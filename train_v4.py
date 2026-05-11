"""
train_v4.py — mel_compact_v4 (intermediate size, unknown-class fix)

Changes vs v2 (3.24M MACs, 30 KB):
  - sinc_stride 16→32      : halves mel output frames (246 vs 492)
  - channels (32,48,64,64) : narrower early blocks vs v2's (48,64,64,64)
  - pools (4,4,2,None)     : 4 conv blocks retained (unlike v3's 3)
  Target: ~1.42M MACs, ~22 KB

Changes vs v3 (1.53M MACs, 18 KB):
  - 4 conv blocks instead of 3
  - narrower channels (32 first block instead of 48)

Additional fix (vs v2 and v3):
  - Per-class augmentation gating: time masking + INT8 noise are skipped for
    the 'unknown' class. These augmentations corrupt the distinguishing cues
    that separate unknown from core commands and degrade unknown accuracy.

Output: weights_v4.h  (evaluate with: make run MODEL=weights_v4.h)
"""

import os
import numpy as np
import tensorflow as tf

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

from dataclasses import dataclass
from load_data import load_speech_commands
from nnom import generate_model as nnom_generate_model
from keras import layers, models, regularizers, optimizers, losses, callbacks

# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RunConfig:
    data_dir: str = "/workspace/Desktop/Models/Ziad/dataset"
    orig_sample_rate: int = 16000
    target_sample_rate: int = 8000
    epochs: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    early_stop_patience: int = 50
    label_smoothing: float = 0.1

CORE_COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
UNKNOWN_LABEL = 'unknown'
CLASSES = sorted(CORE_COMMANDS) + [UNKNOWN_LABEL]
NUM_CLASSES = len(CLASSES)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}
UNKNOWN_IDX = LABEL_TO_IDX[UNKNOWN_LABEL]

config = RunConfig()


# ══════════════════════════════════════════════════════════════════════════
# Mel filterbank initializer (unchanged from baseline)
# ══════════════════════════════════════════════════════════════════════════
def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def _mel_spaced_hz(n_filters, min_hz=50.0, sample_rate=8000):
    nyquist = sample_rate / 2.0
    low_mel  = _hz_to_mel(min_hz)
    high_mel = _hz_to_mel(nyquist)
    mel_pts  = np.linspace(low_mel, high_mel, n_filters + 1)
    hz_pts   = _mel_to_hz(mel_pts)
    centers    = hz_pts[:-1].astype(np.float32)
    bandwidths = np.diff(hz_pts).astype(np.float32)
    return centers, bandwidths

@tf.keras.utils.register_keras_serializable(package='RustPlayground')
class MelBandpassInitializer(tf.keras.initializers.Initializer):
    def __init__(self, n_filters=32, kernel_size=129, sample_rate=8000, min_hz=50.0):
        self.n_filters   = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_hz      = min_hz

    def _build_kernel(self):
        centers, bandwidths = _mel_spaced_hz(
            self.n_filters, min_hz=self.min_hz, sample_rate=self.sample_rate)
        half     = (self.kernel_size - 1) / 2.0
        n        = np.arange(0, self.kernel_size, dtype=np.float32) - half
        n_scaled = n / float(self.sample_rate)
        window   = (0.54 - 0.46 * np.cos(
            2.0 * np.pi * np.arange(self.kernel_size) / (self.kernel_size - 1)
        )).astype(np.float32)
        filters  = np.zeros((self.kernel_size, self.n_filters), dtype=np.float32)
        for i in range(self.n_filters):
            f_low  = max(centers[i], self.min_hz)
            f_high = f_low + max(bandwidths[i], self.min_hz)
            f_high = min(f_high, self.sample_rate / 2.0 - 1.0)
            with np.errstate(divide='ignore', invalid='ignore'):
                lp = np.where(
                    np.abs(n_scaled) < 1e-7,
                    np.float32(2.0 * f_low / self.sample_rate),
                    2.0 * f_low * np.sin(np.pi * 2.0 * f_low * n_scaled)
                    / (np.pi * 2.0 * f_low * n_scaled)
                )
                hp = np.where(
                    np.abs(n_scaled) < 1e-7,
                    np.float32(2.0 * f_high / self.sample_rate),
                    2.0 * f_high * np.sin(np.pi * 2.0 * f_high * n_scaled)
                    / (np.pi * 2.0 * f_high * n_scaled)
                )
            bp = (hp - lp) * window
            filters[:, i] = bp / (np.sum(np.abs(bp)) + 1e-8)
        return filters[:, np.newaxis, :]

    def __call__(self, shape, dtype=None):
        return tf.constant(self._build_kernel(), dtype=dtype or tf.float32)

    def get_config(self):
        return {
            'n_filters': self.n_filters, 'kernel_size': self.kernel_size,
            'sample_rate': self.sample_rate, 'min_hz': self.min_hz,
        }

@tf.keras.utils.register_keras_serializable(package='RustPlayground')
class SmoothnessRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, weight=1e-4):
        self.weight = weight

    def __call__(self, kernel):
        diff = kernel[1:, :, :] - kernel[:-1, :, :]
        return self.weight * tf.reduce_sum(tf.abs(diff))

    def get_config(self):
        return {'weight': self.weight}


# ══════════════════════════════════════════════════════════════════════════
# Augmentation — time masking and INT8 noise gated off for unknown class
# ══════════════════════════════════════════════════════════════════════════
def augment_waveform(x, y):
    is_unknown = tf.equal(y, UNKNOWN_IDX)

    # Wide gain — covers quiet recordings that confuse 'up'/'on'
    if tf.random.uniform([]) < 0.9:
        x = x * tf.random.uniform([], 0.05, 2.0)

    # Time shift
    if tf.random.uniform([]) < 0.8:
        x = tf.roll(x, tf.random.uniform([], -400, 400, dtype=tf.int32), axis=0)

    # Additive noise
    if tf.random.uniform([]) < 0.7:
        x = x + tf.random.normal(tf.shape(x),
                                  stddev=tf.random.uniform([], 0.0, 0.07))

    # SpecAugment-style time masking: skip for unknown (destroys its identity cues)
    if tf.random.uniform([]) < 0.5:
        mask_len   = tf.random.uniform([], 50, 800, dtype=tf.int32)
        mask_start = tf.random.uniform([], 0, 8000 - mask_len, dtype=tf.int32)
        indices    = tf.range(8000, dtype=tf.int32)
        keep = tf.cast(
            tf.logical_or(indices < mask_start, indices >= mask_start + mask_len),
            tf.float32
        )
        x = tf.cond(is_unknown, lambda: x, lambda: x * keep[:, tf.newaxis])

    # INT8 quantization noise: skip for unknown (same reason)
    if tf.random.uniform([]) < 0.3:
        x = tf.cond(
            is_unknown,
            lambda: x,
            lambda: tf.round(x * 128.0) / 128.0
        )

    # Polarity flip
    if tf.random.uniform([]) < 0.2:
        x = -x

    # Soft clip
    if tf.random.uniform([]) < 0.3:
        clip_val = tf.random.uniform([], 0.6, 1.0)
        x = tf.clip_by_value(x, -clip_val, clip_val)

    return tf.clip_by_value(x, -1.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════
# Model — layer names follow NNoM naming rules exactly
#   conv1d_*        → NNoM conv2d_s
#   batch_normalization_* → skipped (folded into conv)
#   re_lu_*         → NNoM act_adv_relu (ReLU6) or act_relu
#   max_pooling_*   → NNoM maxpool_s
#   global_average_pooling → NNoM global_avgpool_s
#   dense_*         → NNoM dense_s
#   dropout         → skipped
#   softmax         → NNoM softmax_s
# ══════════════════════════════════════════════════════════════════════════
def build_mel_compact_v4(
    n_filters=32, kernel_size=129, sinc_stride=32,
    channels=(32, 48, 64, 64), pools=(4, 4, 2, None), fc=64,
    dropout=0.35, smoothness_weight=1e-4, l2_weight=1e-4,
    num_classes=11, input_length=8000,
    name='mel_compact_v4',
):
    l2         = regularizers.l2(l2_weight)
    smooth_reg = SmoothnessRegularizer(weight=smoothness_weight)
    init       = MelBandpassInitializer(n_filters, kernel_size, sample_rate=8000)
    inp = layers.Input(shape=(input_length, 1), name='audio_input')

    # Sinc-like mel filterbank
    x = layers.Conv1D(
        n_filters, kernel_size, strides=sinc_stride, padding='valid',
        use_bias=True, kernel_initializer=init, kernel_regularizer=smooth_reg,
        name='conv1d_mel'
    )(inp)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.ReLU(max_value=6.0, name='re_lu_1')(x)   # ReLU6

    for i, (ch, pool) in enumerate(zip(channels, pools), start=2):
        if pool:
            x = layers.MaxPooling1D(pool, name=f'max_pooling_{i}')(x)
        x = layers.Conv1D(
            ch, 3, padding='same', use_bias=True,
            kernel_regularizer=l2, name=f'conv1d_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'batch_normalization_{i}')(x)
        x = layers.ReLU(max_value=6.0, name=f're_lu_{i}')(x)   # ReLU6

    x   = layers.GlobalAveragePooling1D(name='global_average_pooling')(x)
    x   = layers.Dense(fc, kernel_regularizer=l2, name='dense_fc')(x)
    x   = layers.ReLU(max_value=6.0, name='re_lu_fc')(x)        # ReLU6
    x   = layers.Dropout(dropout, name='dropout')(x)
    out = layers.Dense(num_classes, name='dense_out')(x)
    out = layers.Softmax(name='softmax')(out)

    return models.Model(inp, out, name=name)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 1. Load data
    print("Loading data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_speech_commands(
        DATA_DIR=config.data_dir,
        CORE_COMMANDS=CORE_COMMANDS,
        NUM_CLASSES=NUM_CLASSES,
        LABEL_TO_IDX=LABEL_TO_IDX,
        TARGET_SAMPLE_RATE=config.target_sample_rate,
        ORIG_SAMPLE_RATE=config.orig_sample_rate,
        UNKNOWN_LABEL=UNKNOWN_LABEL,
    )

    # 2. tf.data pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(lambda x, y: (augment_waveform(x, y), y),
             num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(4096)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds  = tf.data.Dataset.from_tensor_slices((x_val,  y_val )).batch(config.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config.batch_size)

    # 3. Class distribution
    print("\nClass distribution — train:")
    for idx, cnt in zip(*np.unique(y_train, return_counts=True)):
        print(f"  {CLASSES[idx]:>8s}: {cnt}")

    # 4. Build model
    model = build_mel_compact_v4()
    model.summary()

    # 5. LR schedule: cosine restarts every 100 epochs
    steps_per_epoch = len(x_train) // config.batch_size
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=config.learning_rate,
        first_decay_steps=100 * steps_per_epoch,
        t_mul=1.0,
        m_mul=0.9,   # each restart the peak LR drops 10%
        alpha=1e-6,
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # 6. Train
    model_path = 'mel_compact_v4_best.h5'
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=config.early_stop_patience,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_path, monitor='val_accuracy', save_best_only=True, verbose=1
        ),
    ]

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=cb_list,
        verbose=2
    )

    # 7. Evaluate
    print("\nEvaluating on test set...")
    _, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Float Test Accuracy: {test_acc*100:.2f}%")

    # 8. Export NNoM — use test + first 500 val samples for calibration
    print("\nExporting to NNoM...")
    x_calib = np.concatenate([x_test, x_val[:500]], axis=0)
    nnom_generate_model(
        model=model,
        x_test=x_calib,
        name='weights_v4.h',
        quantize_method='kld',
        per_channel_quant=True
    )

    print(f"\nFloat accuracy: {test_acc*100:.2f}%")
    print("Generated weights_v4.h")
    print("Evaluate on Spike with:  make run MODEL=weights_v4.h")
