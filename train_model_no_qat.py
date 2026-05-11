import numpy as np
import tensorflow as tf

from dataclasses import dataclass
from load_data import load_speech_commands
from nnom import generate_model as nnom_generate_model
from keras import layers, models, regularizers, optimizers, losses, callbacks

# ══════════════════════════════════════════════════════════════════════════
# Shared helpers & Config
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RunConfig:
    data_dir: str = "/workspace/Desktop/Main/dataset"
    orig_sample_rate: int = 16000
    target_sample_rate: int = 8000
    
    # Set your desired data type here. The augments will scale automatically!
    dtype: type = np.float32 
    
    epochs: int = 500  
    batch_size: int = 128      # Increased to 32 to stabilize training gradients
    learning_rate: float = 1e-3
    early_stop_patience: int = 50
    lr_reduce_patience: int = 20
    lr_reduce_factor: float = 0.5
    min_lr: float = 1e-6

CORE_COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
UNKNOWN_LABEL = 'unknown'
CLASSES = sorted(CORE_COMMANDS) + [UNKNOWN_LABEL]
NUM_CLASSES = len(CLASSES)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}

config = RunConfig()

# ══════════════════════════════════════════════════════════════════════════
# Mel Processing & Initializers
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
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'sample_rate': self.sample_rate,
            'min_hz': self.min_hz,
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
# Dtype-Aware Augmentation Factory
# ══════════════════════════════════════════════════════════════════════════
def get_augmentation_pipeline(dtype=np.float32):
    """
    Returns an augmentation function scaled perfectly to the target data type.
    Toned down significantly to prevent chaotic validation scores.
    """
    if dtype == np.float32:
        max_val, noise_max = 1.0, 0.015 
    elif dtype == np.int16:
        max_val, noise_max = 32767.0, 450.0
    elif dtype == np.int8:
        max_val, noise_max = 127.0, 2.0
    else:
        max_val, noise_max = 1.0, 0.015

    # Ensure dtype cast for TF
    max_val = tf.cast(max_val, tf.float32)
    noise_max = tf.cast(noise_max, tf.float32)

    @tf.function
    def augment_waveform(x):
        # We perform math in float32 to prevent integer overflow during augment
        x_float = tf.cast(x, tf.float32)

        # 1. Random Gain (Toned down to 30% chance, mild adjustments)
        if tf.random.uniform([]) < 0.3:
            gain = tf.random.uniform([], 0.85, 1.15)
            x_float = x_float * gain

        # 2. Time Shift WITH ZERO PADDING (Toned down to 40% chance)
        if tf.random.uniform([]) < 0.4:
            shift = tf.random.uniform([], -200, 200, dtype=tf.int32)
            x_float = tf.roll(x_float, shift=shift, axis=0)
            
            # Mask out the wrapped-around audio with zeros so it sounds natural
            seq_len = tf.shape(x_float)[0]
            mask = tf.range(seq_len)
            if shift > 0:
                valid_mask = tf.cast(mask >= shift, tf.float32)
            else:
                valid_mask = tf.cast(mask < seq_len + shift, tf.float32)
                
            x_float = x_float * tf.expand_dims(valid_mask, -1)

        # 3. Additive Noise (Toned down to 30% chance, lower amplitude)
        if tf.random.uniform([]) < 0.3:
            noise_amp = tf.random.uniform([], 0.0, noise_max)
            noise = tf.random.normal(tf.shape(x_float), stddev=noise_amp)
            x_float = x_float + noise

        # 4. Final Hard Clamp
        x_float = tf.clip_by_value(x_float, -max_val, max_val)

        # Cast back to original dtype
        if dtype in [np.int16, np.int8]:
            return tf.cast(tf.round(x_float), tf.keras.backend.floatx() if dtype == np.float32 else tf.int32)
        return x_float

    return augment_waveform

# ══════════════════════════════════════════════════════════════════════════
# Model Architecture
# ══════════════════════════════════════════════════════════════════════════
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
    'mel_compact': lambda model_name='mel_compact', **kw: build_mel_compact(
        name=model_name, **kw
    ),
    'mel_compact_4blk_ch36': lambda model_name='mel_compact_4blk_ch36', **kw: build_mel_compact(
        channels=(36, 36, 36, 36), pools=(4, 4, 2, None),
        name=model_name, **kw
    ),
}

# ══════════════════════════════════════════════════════════════════════════
# Main Execution Pipeline
# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # 1. Load Data
    print("Loading data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_speech_commands(
        DATA_DIR=config.data_dir,
        CORE_COMMANDS=CORE_COMMANDS,
        NUM_CLASSES=NUM_CLASSES,
        LABEL_TO_IDX=LABEL_TO_IDX,
        TARGET_SAMPLE_RATE=config.target_sample_rate,
        ORIG_SAMPLE_RATE=config.orig_sample_rate,
        UNKNOWN_LABEL=UNKNOWN_LABEL,
        dtype=config.dtype 
    )
    
    # Generate the proper augmentation function for our dtype
    augment_fn = get_augmentation_pipeline(dtype=config.dtype)

    # Convert datasets (Back to simple Sparse mappings)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(
        lambda x, y: (augment_fn(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.shuffle(4096)
    train_ds = train_ds.batch(config.batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(config.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(config.batch_size)

    # Print distribution
    print("\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"  {CLASSES[idx]}: {count} samples")

    # 2. Build Model
    model_name = 'mel_compact_4blk_ch36'
    model = MODEL_BUILDERS[model_name]()
    model.summary()

    # 3. Compile Model (Back to SparseCategoricalCrossentropy)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.learning_rate),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # 4. Define Callbacks
    model_path = f"{model_name}_best.h5"
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

    # 5. Train Model
    # weights = {i: 1.0 for i in range(NUM_CLASSES)}
    # weights[LABEL_TO_IDX['unknown']] = 1.25  # <--- Start with 1.25, don't go crazy yet
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=cb_list,
        # class_weight=weights, # <--- Pass it here
        verbose=2
    )
    
    # 6. Evaluate on Test Set
    print("\nEvaluating on test set...")
    _, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    # 7. Generate NNoM C Header
    print("\nExporting to NNoM...")
    nnom_generate_model(
        model=model,
        x_test=x_test,
        name='weights_no_qat_float_new_noise_run9.h',
        quantize_method='kld',
        per_channel_quant=True
    )

    print("\nFloat reference accuracy on stripped graph:"
          f" {test_acc*100:.2f}%")
    print("Generated weights.h — flash to MCU for true int8 accuracy.")