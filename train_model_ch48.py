import os
import numpy as np
import tensorflow as tf

from pathlib import Path
from nnom import evaluate_model
from dataclasses import dataclass
from load_data import load_speech_commands
from pi_data.load_pi_data import load_speech_commands as load_pi_speech_commands
from nnom import generate_model as nnom_generate_model
from keras import layers, models, regularizers, optimizers, losses, callbacks

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RunConfig:
    data_dir: str = "/workspace/Desktop/Models/Ziad/dataset"
    pi_data_dir: str = "/workspace/Desktop/Models/Ziad/FinalModel/Pi_Dataset"
    orig_sample_rate: int = 16000
    target_sample_rate: int = 8000
    epochs_stage1: int = 500
    batch_size: int = 128
    learning_rate: float = 1e-3
    early_stop_patience: int = 50
    lr_reduce_patience: int = 20
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6
    label_smoothing: float = 0.05

CORE_COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
UNKNOWN_LABEL = 'unknown'
CLASSES = sorted(CORE_COMMANDS) + [UNKNOWN_LABEL]
NUM_CLASSES = len(CLASSES)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}

# Pi directories are capitalized — same sorted order, capitalized
# Down=0, Go=1, Left=2, No=3, Off=4, On=5, Right=6, Stop=7, Up=8, Yes=9, Unknown=10
PI_CLASSES = [c.capitalize() for c in CLASSES[:-1]]  # drop 'unknown', Pi has no real unknowns

config = RunConfig()

def peak_normalize_float(x, target=0.75):
    """Scale float32 clip so peak maps to target (~75% of [-1,1]).
    Clips below silence floor (peak < 1e-4) are left untouched."""
    peak = np.max(np.abs(x))
    if peak < 1e-4:
        return x
    return np.clip(x * min(target / peak, 64.0), -1.0, 1.0).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════
def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)

def _mel_spaced_hz(n_filters, min_hz=50.0, sample_rate=8000):
    nyquist = sample_rate / 2.0
    low_mel = _hz_to_mel(min_hz)
    high_mel = _hz_to_mel(nyquist)
    mel_pts = np.linspace(low_mel, high_mel, n_filters + 1)
    hz_pts = _mel_to_hz(mel_pts)
    centers = hz_pts[:-1].astype(np.float32)
    bandwidths = np.diff(hz_pts).astype(np.float32)
    return centers, bandwidths

@tf.keras.utils.register_keras_serializable(package='RustPlayground')
class MelBandpassInitializer(tf.keras.initializers.Initializer):
    def __init__(self, n_filters=32, kernel_size=129, sample_rate=8000, min_hz=50.0):
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_hz = min_hz

    def _build_kernel(self):
        centers, bandwidths = _mel_spaced_hz(
            self.n_filters, min_hz=self.min_hz, sample_rate=self.sample_rate)
        half = (self.kernel_size - 1) / 2.0
        n = np.arange(0, self.kernel_size, dtype=np.float32) - half
        n_scaled = n / float(self.sample_rate)
        window = (0.54 - 0.46 * np.cos(
            2.0 * np.pi * np.arange(self.kernel_size) / (self.kernel_size - 1)
        )).astype(np.float32)
        filters = np.zeros((self.kernel_size, self.n_filters), dtype=np.float32)
        for i in range(self.n_filters):
            f_low = max(centers[i], self.min_hz)
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

def augment_waveform(x):
    if tf.random.uniform([]) < 0.9:
        x = x * tf.random.uniform([], 0.1, 1.5)
    if tf.random.uniform([]) < 0.8:
        x = tf.roll(x, tf.random.uniform([], -400, 400, dtype=tf.int32), axis=0)
    if tf.random.uniform([]) < 0.7:
        x = x + tf.random.normal(tf.shape(x), stddev=tf.random.uniform([], 0.0, 0.07))
    if tf.random.uniform([]) < 0.2:
        x = -x
    if tf.random.uniform([]) < 0.3:
        x = tf.clip_by_value(x, -tf.random.uniform([], 0.6, 1.0),
                                  tf.random.uniform([], 0.6, 1.0))
    return tf.clip_by_value(x, -1.0, 1.0)


def augment_waveform_pi(x):
    """Aggressive augmentation for Pi samples: wide gain, time masking, more noise."""
    # Very wide gain — covers both quiet and loud Pi recordings
    if tf.random.uniform([]) < 0.95:
        x = x * tf.random.uniform([], 0.03, 2.0)
    # Larger time shift
    if tf.random.uniform([]) < 0.9:
        x = tf.roll(x, tf.random.uniform([], -600, 600, dtype=tf.int32), axis=0)
    # More additive noise
    if tf.random.uniform([]) < 0.8:
        x = x + tf.random.normal(tf.shape(x), stddev=tf.random.uniform([], 0.0, 0.12))
    # Polarity flip
    if tf.random.uniform([]) < 0.3:
        x = -x
    # Time masking: zero out a random window (simulates truncated / quiet-start words)
    if tf.random.uniform([]) < 0.5:
        start  = tf.random.uniform([], 0, 7000, dtype=tf.int32)
        length = tf.random.uniform([], 200, 1000, dtype=tf.int32)
        t      = tf.cast(tf.range(8000), tf.int32)
        mask   = tf.cast(tf.logical_or(t < start, t >= start + length), tf.float32)
        x = x * mask[:, tf.newaxis]
    # Soft clip
    if tf.random.uniform([]) < 0.4:
        x = tf.clip_by_value(x, -tf.random.uniform([], 0.5, 1.0),
                                  tf.random.uniform([], 0.5, 1.0))
    return tf.clip_by_value(x, -1.0, 1.0)

def build_mel_compact(
    n_filters=16, kernel_size=65, sinc_stride=16,
    channels=(48, 48, 48), pools=(4, 4, 2), fc=16,
    dropout=0.3, smoothness_weight=1e-4, l2_weight=1e-4,
    num_classes=11, input_length=8000, causal=False,
    name='mel_compact',
):
    l2 = regularizers.l2(l2_weight)
    smooth_reg = SmoothnessRegularizer(weight=smoothness_weight)
    init = MelBandpassInitializer(n_filters, kernel_size, sample_rate=8000)
    inp = layers.Input(shape=(input_length, 1), name='audio_input')

    x = layers.Conv1D(
        n_filters, kernel_size, strides=sinc_stride, padding='valid',
        use_bias=True, kernel_initializer=init, kernel_regularizer=smooth_reg,
        name='conv1d_mel'
    )(inp)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.ReLU(name='re_lu_1')(x)

    conv_padding = 'causal' if causal else 'same'
    for i, (ch, pool) in enumerate(zip(channels, pools), start=2):
        if pool:
            x = layers.MaxPooling1D(pool, name=f'max_pooling_{i}')(x)
        x = layers.Conv1D(
            ch, 3, padding=conv_padding, use_bias=True,
            kernel_regularizer=l2, name=f'conv1d_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'batch_normalization_{i}')(x)
        x = layers.ReLU(name=f're_lu_{i}')(x)

    x = layers.GlobalAveragePooling1D(name='global_average_pooling')(x)
    x = layers.Dense(fc, kernel_regularizer=l2, name='dense_fc')(x)
    x = layers.ReLU(name='re_lu_fc')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    out = layers.Dense(num_classes, name='dense_out')(x)
    out = layers.Softmax(name='softmax')(out)

    return models.Model(inp, out, name=name)

MODEL_BUILDERS = {
    'mel_compact_4blk_ch48': lambda model_name='mel_compact_4blk_ch48', **kw: build_mel_compact(
        channels=(48, 48, 48, 48), pools=(4, 4, 2, None),
        name=model_name, **kw
    ),
}

# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # ── 1. Load GSCD ─────────────────────────────────────────────────────
    print("Loading GSCD data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_speech_commands(
        DATA_DIR=config.data_dir,
        CORE_COMMANDS=CORE_COMMANDS,
        NUM_CLASSES=NUM_CLASSES,
        LABEL_TO_IDX=LABEL_TO_IDX,
        TARGET_SAMPLE_RATE=config.target_sample_rate,
        ORIG_SAMPLE_RATE=config.orig_sample_rate,
        UNKNOWN_LABEL=UNKNOWN_LABEL,
    )

    # ── 2. Load Pi data (capitalized GSCD-order classes, unknown synthetic) ─
    print("\nLoading Pi data...")
    (x_pi_train, y_pi_train), (_, _), (_, _) = load_pi_speech_commands(
        DATA_DIR=config.pi_data_dir,
        CLASSES=PI_CLASSES,           # ['Down','Go','Left',...,'Yes'] — same index order as GSCD
        TARGET_SAMPLE_RATE=config.target_sample_rate,
        ORIG_SAMPLE_RATE=config.orig_sample_rate,
        val_split=0.1,
        test_split=0.1,
        add_unknown=True,
    )

    # Peak-normalise only real speech clips — synthetic unknowns (label=10) stay at
    # natural low amplitude so the model learns to reject quiet noise, not loud noise.
    PI_UNKNOWN_IDX = len(PI_CLASSES)  # = 10
    x_pi_train = np.array([
        peak_normalize_float(x) if y != PI_UNKNOWN_IDX else x
        for x, y in zip(x_pi_train, y_pi_train)
    ], dtype=np.float32)

    PI_WEIGHT = 10.0  # Pi samples contribute 10x more to the loss (~25% of total)
    print(f"\nGSCD train: {len(x_train)}  Pi train: {len(x_pi_train)}")
    print(f"Pi fraction: {len(x_pi_train) / (len(x_train) + len(x_pi_train)) * 100:.1f}%")
    print(f"Pi loss weight: {PI_WEIGHT}x  (effective Pi loss share: "
          f"{PI_WEIGHT*len(x_pi_train)/(PI_WEIGHT*len(x_pi_train)+len(x_train))*100:.1f}%)")

    # ── 3. Print distribution ─────────────────────────────────────────────
    print("\nMixed train distribution:")
    all_y = np.concatenate([y_train, y_pi_train])
    unique, counts = np.unique(all_y, return_counts=True)
    for i, c in zip(unique, counts):
        print(f"  {CLASSES[i]}: {c}")

    # ── 4. Build tf.data pipelines ─────────────────────────────────────────
    # GSCD: standard augmentation, weight 1.0
    with tf.device('/cpu:0'):
        gscd_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .map(lambda x, y: (augment_waveform(x), y, tf.constant(1.0)),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(4096)
        )
        # Pi: aggressive augmentation, weight PI_WEIGHT
        pi_ds = (
            tf.data.Dataset.from_tensor_slices((x_pi_train, y_pi_train))
            .map(lambda x, y: (augment_waveform_pi(x), y, tf.constant(PI_WEIGHT)),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(1024)
        )
    train_ds = (
        gscd_ds.concatenate(pi_ds)
        .shuffle(5120)
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(config.batch_size)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(config.batch_size)
    )

    # ── 6. Build & compile ────────────────────────────────────────────────
    model_name = 'mel_compact_4blk_ch48'
    model = MODEL_BUILDERS[model_name]()
    model.summary()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.learning_rate),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # ── 7. Train ──────────────────────────────────────────────────────────
    model_path = f"{model_name}_pi_mix_norm_best.h5"
    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=config.early_stop_patience,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_path, monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=config.lr_reduce_factor,
            patience=config.lr_reduce_patience, min_lr=config.min_lr, verbose=1
        )
    ]

    print("\nStarting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs_stage1,
        callbacks=cb_list,
        verbose=2
    )

    # ── 8. Evaluate on GSCD test set ─────────────────────────────────────
    _, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nGSCD Test Accuracy (float): {test_acc*100:.2f}%")

    # ── 9. Export NNoM weights (calibrated on GSCD test set) ─────────────
    print("\nExporting to NNoM...")
    nnom_generate_model(
        model=model,
        x_test=x_test,
        name='weights_ch48_pi_mix_norm.h',
        quantize_method='kld',
        per_channel_quant=True
    )

    print("\nDone. Weights written to weights_ch48_pi_mix_norm.h")
    print("Run Pi test: make run_pi MODEL=weights_ch48_pi_mix_norm.h")
