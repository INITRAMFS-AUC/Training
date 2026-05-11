"""
train_multi_cmd.py — Parameterized copy of train_4cmd.py.

Runs the exact same pipeline for any subset of:
  down, go, left, no, off, on, right, stop, up, yes

Usage:
  python train_multi_cmd.py --commands go left right stop
  python train_multi_cmd.py --commands up down left right
  python train_multi_cmd.py --commands on off

Differences from train_4cmd.py (logic is identical, only these changed):
  1. --commands / --out-dir CLI args instead of hardcoded CORE_COMMANDS
  2. GSCD_DIR corrected to /workspace/Ziad/dataset   (Desktop path no longer exists)
  3. PI_DIR corrected to Pi_Dataset/                  (Desktop path no longer exists;
     Pi_Dataset has same 16kHz int16 format, capitalized subfolder names, .wav extension)
  4. Output filenames derived from the command set
"""

import os, math, argparse
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
for _gpu in tf.config.list_physical_devices('GPU'):
    try: tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError: pass

# ── parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--commands', nargs='+', required=True)
parser.add_argument('--out-dir', default=None)
args = parser.parse_args()

# ── paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CKPT_IN      = os.path.join(_SCRIPT_DIR, 'mel_compact_4blk_ch36_int8norm.h5')
BARBARY_DIR  = os.path.join(_SCRIPT_DIR, 'new_dataset_test')   # full superset dataset
GSCD_DIR     = '/workspace/Ziad/dataset'
PI_DIR       = os.path.join(_SCRIPT_DIR, 'Pi_Dataset')   # capitalized subfolders, plain .wav

# ── label map ─────────────────────────────────────────────────────────────────
CORE_COMMANDS = sorted(args.commands)
USE_CASE      = '_'.join(CORE_COMMANDS)
UNKNOWN_LABEL = 'unknown'
CLASSES       = CORE_COMMANDS + [UNKNOWN_LABEL]
NUM_CLASSES   = len(CLASSES)
LABEL_TO_IDX  = {lbl: i for i, lbl in enumerate(CLASSES)}
FOLDER_TO_LABEL = {cmd.title(): cmd for cmd in CORE_COMMANDS}

OUT_DIR = args.out_dir or os.path.join(_SCRIPT_DIR, 'multi_cmd_results', USE_CASE)
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_OUT    = os.path.join(OUT_DIR, f'mel_compact_{USE_CASE}_int8norm.h5')
WEIGHTS_OUT = os.path.join(OUT_DIR, f'weights_{USE_CASE}.h')

print(f"\n{'='*60}")
print(f"USE CASE : {USE_CASE}")
print(f"CLASSES  : {CLASSES}")
print(f"OUT DIR  : {OUT_DIR}")
print(f"{'='*60}\n")

from load_data import load_speech_commands
from nnom import generate_model as nnom_generate_model
from keras import optimizers, losses, callbacks

ORIG_SR  = 16000
SAMPLES  = 8000
BATCH    = 64
LR_MAX   = 5e-5
LR_MIN   = 1e-6
EPOCHS   = 150
PATIENCE = 40
BARBARY_OVERSAMPLE = 40

# ── custom objects (must match checkpoint) ────────────────────────────────────
def _mel_to_hz(m): return 700.0*(10.0**(m/2595.0)-1.0)
def _hz_to_mel(f): return 2595.0*np.log10(1.0+f/700.0)
def _mel_spaced_hz(n, min_hz=50.0, sr=8000):
    nyq = sr/2.0
    mel = np.linspace(_hz_to_mel(min_hz), _hz_to_mel(nyq), n+1)
    hz  = _mel_to_hz(mel)
    return hz[:-1].astype(np.float32), np.diff(hz).astype(np.float32)

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

# ── int8 pipeline ─────────────────────────────────────────────────────────────
def int8_peak_norm(x_float, target=96):
    x_int8 = np.clip(np.round(x_float * 128.0), -128, 127).astype(np.int8)
    peak   = int(np.max(np.abs(x_int8.astype(np.int32))))
    if peak < 1:
        return x_float
    scale  = min(target / peak, 64.0)
    x_int8 = np.clip(np.round(x_int8.astype(np.float32) * scale), -128, 127).astype(np.int8)
    return x_int8.astype(np.float32) / 128.0

def apply_int8_pipeline(x):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = int8_peak_norm(x[i])
    return out

# ── augmentation ──────────────────────────────────────────────────────────────
def augment_waveform(x):
    if tf.random.uniform([]) < 0.8:
        x = x * tf.random.uniform([], 0.6, 1.4)
    if tf.random.uniform([]) < 0.8:
        x = tf.roll(x, tf.random.uniform([], -800, 800, dtype=tf.int32), axis=0)
    if tf.random.uniform([]) < 0.7:
        x = x + tf.random.normal(tf.shape(x), stddev=tf.random.uniform([], 0.0, 0.06))
    if tf.random.uniform([]) < 0.3:
        x = -x
    if tf.random.uniform([]) < 0.4:
        x = tf.clip_by_value(x, -tf.random.uniform([], 0.5, 0.95),
                                  tf.random.uniform([], 0.5, 0.95))
    return tf.clip_by_value(x, -1.0, 1.0)

# ── Barbary loader ────────────────────────────────────────────────────────────
def load_barbary_wav(path):
    sr, data = wavfile.read(path)
    if data.ndim > 1: data = data[:,0]
    if data.dtype != np.int16: data = data.astype(np.int16)
    data = data[::2]
    if len(data) > SAMPLES: data = data[:SAMPLES]
    elif len(data) < SAMPLES: data = np.pad(data,(0,SAMPLES-len(data)))
    x_int8 = np.right_shift(data, 8).astype(np.int8)
    peak   = int(np.max(np.abs(x_int8.astype(np.int32))))
    if peak >= 1:
        scale  = min(96.0 / peak, 64.0)
        x_int8 = np.clip(np.round(x_int8.astype(np.float32)*scale), -128, 127).astype(np.int8)
    return x_int8.astype(np.float32) / 128.0

def load_barbary_dataset(rng):
    x_tr, y_tr, x_te, y_te = [], [], [], []
    for folder in sorted(os.listdir(BARBARY_DIR)):
        fpath = os.path.join(BARBARY_DIR, folder)
        if not os.path.isdir(fpath): continue
        label_str = FOLDER_TO_LABEL.get(folder)
        if label_str is None: continue   # skip keywords not in our command set
        label_idx = LABEL_TO_IDX[label_str]
        wavs = sorted(f for f in os.listdir(fpath) if f.endswith('_16k_s16_monitor.wav'))
        if not wavs: continue
        indices = rng.permutation(len(wavs))
        n_test  = max(1, round(len(wavs) * 0.20))   # 20% holdout (same ratio as original 8/40)
        for i, idx in enumerate(indices):
            audio = load_barbary_wav(os.path.join(fpath, wavs[idx]))
            if i < n_test:
                x_te.append(audio); y_te.append(label_idx)
            else:
                x_tr.append(audio); y_tr.append(label_idx)
        print(f"  {folder:8s}: {len(wavs)-n_test} train, {n_test} test")
    return (np.array(x_tr, np.float32), np.array(y_tr, np.int32),
            np.array(x_te, np.float32), np.array(y_te, np.int32))

# ── pi_dataset loader (Pi_Dataset/ — capitalized folders, plain .wav) ─────────
def load_pi_clips(cmd):
    # Pi_Dataset uses Title-case subfolder names and plain .wav files
    folder = os.path.join(PI_DIR, cmd.title())
    clips, labels = [], []
    if not os.path.isdir(folder):
        return clips, labels
    for f in sorted(os.listdir(folder)):
        if not f.endswith('.wav'): continue
        sr, data = wavfile.read(os.path.join(folder, f))
        if data.ndim > 1: data = data[:,0]
        if data.dtype != np.int16: data = data.astype(np.int16)
        data = data[::2]
        if len(data) > SAMPLES: data = data[:SAMPLES]
        elif len(data) < SAMPLES: data = np.pad(data,(0,SAMPLES-len(data)))
        x_int8 = np.right_shift(data, 8).astype(np.int8)
        peak   = int(np.max(np.abs(x_int8.astype(np.int32))))
        if peak >= 1:
            scale  = min(96.0 / peak, 64.0)
            x_int8 = np.clip(np.round(x_int8.astype(np.float32)*scale), -128, 127).astype(np.int8)
        clips.append(x_int8.astype(np.float32) / 128.0)
        labels.append(LABEL_TO_IDX[cmd])
    return clips, labels

# ── cosine LR schedule ────────────────────────────────────────────────────────
class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_max, lr_min, total_steps):
        self.lr_max=float(lr_max); self.lr_min=float(lr_min)
        self.total_steps=float(total_steps)
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.lr_min + 0.5*(self.lr_max-self.lr_min)*(
            1.0+tf.cos(math.pi*step/self.total_steps))
    def get_config(self):
        return {'lr_max':self.lr_max,'lr_min':self.lr_min,'total_steps':self.total_steps}

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rng = np.random.default_rng(42)

    # 1. Load GSCD filtered to our commands
    print(f"Loading GSCD ({USE_CASE})...")
    (x_gscd_tr, y_gscd_tr), (x_val, y_val), (x_gscd_te, y_gscd_te) = load_speech_commands(
        DATA_DIR=GSCD_DIR, CORE_COMMANDS=CORE_COMMANDS,
        NUM_CLASSES=NUM_CLASSES, LABEL_TO_IDX=LABEL_TO_IDX,
        TARGET_SAMPLE_RATE=8000, ORIG_SAMPLE_RATE=16000,
        UNKNOWN_LABEL=UNKNOWN_LABEL,
    )
    x_gscd_tr = x_gscd_tr.reshape(-1, SAMPLES).astype(np.float32)
    x_val     = x_val.reshape(-1, SAMPLES).astype(np.float32)
    x_gscd_te = x_gscd_te.reshape(-1, SAMPLES).astype(np.float32)

    print("Applying int8 pipeline to GSCD...")
    x_gscd_tr = apply_int8_pipeline(x_gscd_tr)
    x_val     = apply_int8_pipeline(x_val)
    x_gscd_te = apply_int8_pipeline(x_gscd_te)

    # 2. Load Barbary (only folders matching our commands)
    print("\nLoading Barbary dataset...")
    x_b_tr, y_b_tr, x_b_te, y_b_te = load_barbary_dataset(rng)

    # 3. Mix
    if len(x_b_tr) > 0:
        x_b_rep = np.tile(x_b_tr, (BARBARY_OVERSAMPLE, 1))
        y_b_rep = np.tile(y_b_tr,  BARBARY_OVERSAMPLE)
        x_train = np.concatenate([x_gscd_tr, x_b_rep], axis=0)
        y_train = np.concatenate([y_gscd_tr, y_b_rep], axis=0)
        print(f"\nMixed: {len(x_gscd_tr)} GSCD + {len(x_b_rep)} Barbary = {len(x_train)} total")
    else:
        x_train = x_gscd_tr
        y_train = y_gscd_tr
        x_b_te  = np.empty((0, SAMPLES), np.float32)
        y_b_te  = np.empty((0,),         np.int32)
        print(f"\nNo Barbary data for this use case — using {len(x_train)} GSCD samples only")

    x_train   = x_train[:, :, np.newaxis]
    x_val     = x_val[:, :, np.newaxis]
    x_gscd_te = x_gscd_te[:, :, np.newaxis]
    x_b_te    = x_b_te[:, :, np.newaxis]

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .map(lambda x,y: (augment_waveform(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(16384).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    )
    val_ds     = tf.data.Dataset.from_tensor_slices((x_val,     y_val    )).batch(BATCH)
    gscd_te_ds = tf.data.Dataset.from_tensor_slices((x_gscd_te, y_gscd_te)).batch(BATCH)
    barb_te_ds = tf.data.Dataset.from_tensor_slices((x_b_te,    y_b_te   )).batch(BATCH)

    # 4. Build N-class model from 11-class backbone
    print(f"\nLoading backbone: {CKPT_IN}")
    base = tf.keras.models.load_model(
        CKPT_IN,
        custom_objects={'MelBandpassInitializer': MelBandpassInitializer,
                        'SmoothnessRegularizer':  SmoothnessRegularizer},
        compile=False,
    )
    backbone_out = base.get_layer('dropout').output
    x = tf.keras.layers.Dense(NUM_CLASSES, name=f'dense_out_{USE_CASE}')(backbone_out)
    x = tf.keras.layers.Softmax(name=f'softmax_{USE_CASE}')(x)
    model = tf.keras.Model(inputs=base.input, outputs=x)

    # 5. Phase 1: freeze backbone, train only new head
    print("\nPhase 1: train head only (20 epochs)...")
    for layer in model.layers:
        layer.trainable = (layer.name in (f'dense_out_{USE_CASE}', f'softmax_{USE_CASE}'))
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=20, verbose=2)

    # 6. Phase 2: unfreeze all, fine-tune end-to-end
    print("\nPhase 2: fine-tune all layers...")
    for layer in model.layers:
        layer.trainable = True
    steps_per_epoch = math.ceil(len(x_train) / BATCH)
    total_steps     = EPOCHS * steps_per_epoch
    opt = optimizers.Adam(learning_rate=CosineSchedule(LR_MAX, LR_MIN, total_steps))
    model.compile(
        optimizer=opt,
        loss=losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
    )
    cb_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=PATIENCE,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(CKPT_OUT, monitor='val_accuracy',
                                  save_best_only=True, verbose=1),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              callbacks=cb_list, verbose=2)

    # 7. Evaluate
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    _, acc_gscd = model.evaluate(gscd_te_ds, verbose=0)
    print(f"\nFine-tuned — GSCD test: {acc_gscd*100:.2f}%")

    if len(x_b_te) > 0:
        _, acc_barb = model.evaluate(barb_te_ds, verbose=0)
        print(f"Fine-tuned — Barbary:   {acc_barb*100:.2f}%")
        print("\nPer-class on Barbary test:")
        preds_b = np.argmax(model.predict(barb_te_ds, verbose=0), axis=1)
        for i, cls in enumerate(CLASSES):
            mask = y_b_te == i
            if mask.sum() == 0: continue
            print(f"  {cls:8s}: {np.mean(preds_b[mask]==i)*100:.1f}%  ({mask.sum()} samples)")

    print("\nPer-class on GSCD test:")
    preds_g = np.argmax(model.predict(gscd_te_ds, verbose=0), axis=1)
    for i, cls in enumerate(CLASSES):
        mask = y_gscd_te == i
        if mask.sum() == 0: continue
        print(f"  {cls:8s}: {np.mean(preds_g[mask]==i)*100:.1f}%  ({mask.sum()} samples)")

    # 8. Pi dataset eval + calibration
    print("\nLoading pi_dataset for eval + calibration...")
    pi_clips, pi_labels = [], []
    for cmd in CORE_COMMANDS:
        c, l = load_pi_clips(cmd)
        pi_clips.extend(c); pi_labels.extend(l)
        print(f"  {cmd:8s}: {len(c)} clips")

    x_pi = np.array(pi_clips, np.float32)[:, :, np.newaxis]
    y_pi = np.array(pi_labels, np.int32)
    pi_ds = tf.data.Dataset.from_tensor_slices((x_pi, y_pi)).batch(BATCH)

    _, acc_pi = model.evaluate(pi_ds, verbose=0)
    preds_pi  = np.argmax(model.predict(pi_ds, verbose=0), axis=1)
    print(f"\nPi dataset overall: {acc_pi*100:.2f}%")
    for i, cls in enumerate(CLASSES):
        mask = y_pi == i
        if mask.sum() == 0: continue
        print(f"  {cls:8s}: {np.mean(preds_pi[mask]==i)*100:.1f}%  ({mask.sum()} clips)")

    # 9. PTQ export — KLD calibration
    x_calib = np.concatenate([
        x_gscd_te[:, :, 0],
        x_val[:500, :, 0],
        x_b_te[:, :, 0],
        x_pi[:, :, 0],
    ], axis=0)
    x_calib = x_calib[:, :, np.newaxis]
    x_calib = x_calib[rng.permutation(len(x_calib))]

    print(f"\nExporting PTQ ({len(x_calib)} calibration samples)...")
    nnom_generate_model(model=model, x_test=x_calib,
                        name=WEIGHTS_OUT, quantize_method='kld', per_channel_quant=True)
    print(f"\nDone.")
    print(f"  Model  : {CKPT_OUT}")
    print(f"  Weights: {WEIGHTS_OUT}")
    print(f"  Classes: {CLASSES}")
