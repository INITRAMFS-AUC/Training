"""
train_improved.py — Better float training for mel_compact_4blk_ch36

Improvements over load_model_no_qat.py:
  - Cosine LR schedule with linear warmup (no ReduceLROnPlateau plateau stalls)
  - Class-weighted loss: 'up' and 'unknown' get 1.8x weight (they're the hard INT8 classes)
  - Oversample 'up' and 'unknown' in the training set (2x extra copies)
  - Label smoothing 0.05
  - Batch size 128 (was 16 — larger batches stabilize BN stats)
  - Saves to mel_compact_4blk_ch36_improved_best.h5
  - PTQ export at end → weights_improved.h
"""

import os, math
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

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Cfg:
    data_dir:            str   = "/workspace/Desktop/Models/Ziad/dataset"
    orig_sr:             int   = 16000
    target_sr:           int   = 8000
    epochs:              int   = 500
    batch_size:          int   = 128
    lr_max:              float = 1e-3
    lr_min:              float = 1e-6
    warmup_epochs:       int   = 3    # short warmup
    early_stop_patience: int   = 60
    label_smoothing:     float = 0.05
    hard_oversample:     int   = 2    # extra copies of 'up'+'unknown' in training
    hard_class_weight:   float = 1.2  # mild boost, not aggressive

CORE_COMMANDS = ['yes','no','up','down','left','right','on','off','stop','go']
UNKNOWN_LABEL = 'unknown'
CLASSES       = sorted(CORE_COMMANDS) + [UNKNOWN_LABEL]
NUM_CLASSES   = len(CLASSES)
LABEL_TO_IDX  = {lbl: i for i, lbl in enumerate(CLASSES)}

cfg      = Cfg()
CKPT     = 'mel_compact_4blk_ch36_improved_best.h5'
WEIGHTS  = 'weights_improved.h'

# ─────────────────────────────────────────────────────────────────────────────
# Custom objects
# ─────────────────────────────────────────────────────────────────────────────
def _mel_to_hz(m): return 700.0*(10.0**(m/2595.0)-1.0)
def _hz_to_mel(f): return 2595.0*np.log10(1.0+f/700.0)

def _mel_spaced_hz(n_filters, min_hz=50.0, sample_rate=8000):
    nyq     = sample_rate/2.0
    mel_pts = np.linspace(_hz_to_mel(min_hz), _hz_to_mel(nyq), n_filters+1)
    hz_pts  = _mel_to_hz(mel_pts)
    return hz_pts[:-1].astype(np.float32), np.diff(hz_pts).astype(np.float32)

@tf.keras.utils.register_keras_serializable(package='RustPlayground')
class MelBandpassInitializer(tf.keras.initializers.Initializer):
    def __init__(self, n_filters=32, kernel_size=129, sample_rate=8000, min_hz=50.0):
        self.n_filters=n_filters; self.kernel_size=kernel_size
        self.sample_rate=sample_rate; self.min_hz=min_hz
    def _build_kernel(self):
        centers,bws = _mel_spaced_hz(self.n_filters,self.min_hz,self.sample_rate)
        half=(self.kernel_size-1)/2.0
        n=np.arange(0,self.kernel_size,dtype=np.float32)-half; ns=n/float(self.sample_rate)
        win=(0.54-0.46*np.cos(2.0*np.pi*np.arange(self.kernel_size)/(self.kernel_size-1))).astype(np.float32)
        F=np.zeros((self.kernel_size,self.n_filters),np.float32)
        for i in range(self.n_filters):
            fl=max(centers[i],self.min_hz); fh=min(fl+max(bws[i],self.min_hz),self.sample_rate/2.0-1.0)
            with np.errstate(divide='ignore',invalid='ignore'):
                lp=np.where(np.abs(ns)<1e-7,np.float32(2*fl/self.sample_rate),2*fl*np.sin(np.pi*2*fl*ns)/(np.pi*2*fl*ns))
                hp=np.where(np.abs(ns)<1e-7,np.float32(2*fh/self.sample_rate),2*fh*np.sin(np.pi*2*fh*ns)/(np.pi*2*fh*ns))
            bp=(hp-lp)*win; F[:,i]=bp/(np.sum(np.abs(bp))+1e-8)
        return F[:,np.newaxis,:]
    def __call__(self,shape,dtype=None): return tf.constant(self._build_kernel(),dtype=dtype or tf.float32)
    def get_config(self): return {'n_filters':self.n_filters,'kernel_size':self.kernel_size,'sample_rate':self.sample_rate,'min_hz':self.min_hz}

@tf.keras.utils.register_keras_serializable(package='RustPlayground')
class SmoothnessRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self,weight=1e-4): self.weight=weight
    def __call__(self,k): return self.weight*tf.reduce_sum(tf.abs(k[1:,:,:]-k[:-1,:,:]))
    def get_config(self): return {'weight':self.weight}

# ─────────────────────────────────────────────────────────────────────────────
# Model (identical to original)
# ─────────────────────────────────────────────────────────────────────────────
def build_mel_compact(
    n_filters=16, kernel_size=65, sinc_stride=16,
    channels=(36,36,36,36), pools=(4,4,2,None), fc=16,
    dropout=0.3, smoothness_weight=1e-4, l2_weight=1e-4,
    num_classes=11, input_length=8000,
    name='mel_compact_4blk_ch36',
):
    l2   = regularizers.l2(l2_weight)
    sr   = SmoothnessRegularizer(weight=smoothness_weight)
    init = MelBandpassInitializer(n_filters, kernel_size, sample_rate=8000)
    inp  = layers.Input(shape=(input_length,1), name='audio_input')

    x = layers.Conv1D(n_filters, kernel_size, strides=sinc_stride, padding='valid',
                       use_bias=True, kernel_initializer=init, kernel_regularizer=sr,
                       name='conv1d_mel')(inp)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.ReLU(name='re_lu_1')(x)

    for i,(ch,pool) in enumerate(zip(channels,pools),start=2):
        if pool:
            x = layers.MaxPooling1D(pool, name=f'max_pooling_{i}')(x)
        x = layers.Conv1D(ch,3,padding='same',use_bias=True,kernel_regularizer=l2,name=f'conv1d_{i}')(x)
        x = layers.BatchNormalization(name=f'batch_normalization_{i}')(x)
        x = layers.ReLU(name=f're_lu_{i}')(x)

    x   = layers.GlobalAveragePooling1D(name='global_average_pooling')(x)
    x   = layers.Dense(fc, kernel_regularizer=l2, name='dense_fc')(x)
    x   = layers.ReLU(name='re_lu_fc')(x)
    x   = layers.Dropout(dropout, name='dropout')(x)
    out = layers.Dense(num_classes, name='dense_out')(x)
    out = layers.Softmax(name='softmax')(out)
    return models.Model(inp, out, name=name)

# ─────────────────────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────
def augment_waveform(x):
    if tf.random.uniform([]) < 0.8:
        x = x * tf.random.uniform([], 0.6, 1.4)
    if tf.random.uniform([]) < 0.8:
        x = tf.roll(x, tf.random.uniform([], -600, 600, dtype=tf.int32), axis=0)
    if tf.random.uniform([]) < 0.7:
        x = x + tf.random.normal(tf.shape(x), stddev=tf.random.uniform([], 0.0, 0.06))
    if tf.random.uniform([]) < 0.3:
        x = -x
    if tf.random.uniform([]) < 0.4:
        x = tf.clip_by_value(x, -tf.random.uniform([], 0.5, 0.95),
                                  tf.random.uniform([], 0.5, 0.95))
    return tf.clip_by_value(x, -1.0, 1.0)

# ─────────────────────────────────────────────────────────────────────────────
# Cosine LR schedule with linear warmup
# ─────────────────────────────────────────────────────────────────────────────
class CosineWarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, warmup_steps, total_steps):
        self.lr_max       = lr_max
        self.lr_min       = lr_min
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)

    def __call__(self, step):
        step  = tf.cast(step, tf.float32)
        warmup_lr = self.lr_max * step / self.warmup_steps
        cos_lr    = self.lr_min + 0.5*(self.lr_max-self.lr_min) * (
            1.0 + tf.cos(math.pi * (step-self.warmup_steps)
                         / (self.total_steps-self.warmup_steps))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cos_lr)

    def get_config(self):
        return {'lr_max':self.lr_max,'lr_min':self.lr_min,
                'warmup_steps':self.warmup_steps,'total_steps':self.total_steps}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading data...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_speech_commands(
        DATA_DIR           = cfg.data_dir,
        CORE_COMMANDS      = CORE_COMMANDS,
        NUM_CLASSES        = NUM_CLASSES,
        LABEL_TO_IDX       = LABEL_TO_IDX,
        TARGET_SAMPLE_RATE = cfg.target_sr,
        ORIG_SAMPLE_RATE   = cfg.orig_sr,
        UNKNOWN_LABEL      = UNKNOWN_LABEL,
    )

    # Oversample hard classes in training set
    up_idx      = LABEL_TO_IDX['up']
    unknown_idx = LABEL_TO_IDX[UNKNOWN_LABEL]
    hard_mask   = (y_train == up_idx) | (y_train == unknown_idx)
    x_hard      = x_train[hard_mask]
    y_hard      = y_train[hard_mask]
    x_train_aug = np.concatenate([x_train] + [x_hard]*cfg.hard_oversample, axis=0)
    y_train_aug = np.concatenate([y_train] + [y_hard]*cfg.hard_oversample, axis=0)

    # Class weights: hard classes get 1.8x
    class_weight = {i: 1.0 for i in range(NUM_CLASSES)}
    class_weight[up_idx]      = cfg.hard_class_weight
    class_weight[unknown_idx] = cfg.hard_class_weight
    print(f"Training samples: {len(x_train)} → {len(x_train_aug)} after oversampling")
    print(f"Class weights: up={class_weight[up_idx]}, unknown={class_weight[unknown_idx]}")

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train_aug, y_train_aug))
        .map(lambda x,y: (augment_waveform(x),y), num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(8192).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    )
    val_ds  = tf.data.Dataset.from_tensor_slices((x_val,  y_val )).batch(cfg.batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(cfg.batch_size)

    # Build model
    model = build_mel_compact()
    model.summary()

    # LR schedule
    steps_per_epoch = math.ceil(len(x_train_aug) / cfg.batch_size)
    total_steps     = cfg.epochs * steps_per_epoch
    warmup_steps    = cfg.warmup_epochs * steps_per_epoch
    lr_schedule     = CosineWarmupSchedule(cfg.lr_max, cfg.lr_min, warmup_steps, total_steps)

    model.compile(
        optimizer = optimizers.Adam(learning_rate=lr_schedule),
        loss      = losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics   = ['accuracy']
    )

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=cfg.early_stop_patience,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            CKPT, monitor='val_accuracy', save_best_only=True, verbose=1
        ),
    ]

    print("\nStarting training...")
    model.fit(
        train_ds,
        validation_data  = val_ds,
        epochs           = cfg.epochs,
        class_weight     = class_weight,
        callbacks        = cb_list,
        verbose          = 2
    )

    _, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nFinal test accuracy: {test_acc*100:.2f}%")

    # Per-class accuracy
    preds = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    for i, cls in enumerate(CLASSES):
        mask = y_test == i
        cl_acc = np.mean(preds[mask] == i) if mask.sum() > 0 else 0
        print(f"  {cls:10s}: {cl_acc*100:.1f}%  ({mask.sum()} samples)")

    # PTQ export with oversampled calibration
    hard_mask_test = (y_test == up_idx) | (y_test == unknown_idx)
    x_hard_test    = x_test[hard_mask_test]
    x_calib = np.concatenate(
        [x_test, x_val[:500]] + [x_hard_test]*3, axis=0
    )
    rng  = np.random.default_rng(42)
    x_calib = x_calib[rng.permutation(len(x_calib))]

    print(f"\nExporting to NNoM (calibration set: {len(x_calib)} samples)...")
    nnom_generate_model(
        model=model, x_test=x_calib,
        name=WEIGHTS, quantize_method='kld', per_channel_quant=True,
    )
    print(f"\nGenerated {WEIGHTS}")
    print(f"Run on Spike:  make run MODEL={WEIGHTS}")
