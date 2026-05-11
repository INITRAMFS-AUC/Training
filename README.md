# KWS Production — Best Models & Training Pipeline

Keyword spotting (KWS) system for a custom RISC-V SoC, trained on Google Speech Commands Dataset
(GSCD) and validated on real Raspberry Pi recordings (Pi dataset) and Barbary recordings. Deployed
via NNoM int8 quantized inference on bare-metal firmware.

---

## Architecture: mel_compact_4blk_ch36

A compact, fully-learnable KWS model that fuses feature extraction and classification into a single
end-to-end network. Input: 8000 raw audio samples (1 second at 8 kHz), int8 peak-normalized.

```
Input (8000,1)
│
├── [mel frontend]  Conv1D  k=65, stride=16, 16 filters
│                   Sinc bandpass init (Hamming-windowed, mel-spaced)
│                   SmoothnessRegularizer — penalizes non-smooth filter shape
│                   Output: (500, 16)
│
├── [block 1]  DWConv1D k=9, BN, ReLU → PWConv1D 36ch, BN, ReLU → MaxPool/4
│                   Output: (125, 36)
├── [block 2]  DWConv1D k=9, BN, ReLU → PWConv1D 36ch, BN, ReLU → MaxPool/4
│                   Output: (31, 36)
├── [block 3]  DWConv1D k=9, BN, ReLU → PWConv1D 36ch, BN, ReLU → MaxPool/2
│                   Output: (15, 36)
├── [block 4]  DWConv1D k=9, BN, ReLU → PWConv1D 36ch, BN, ReLU → GlobalAvgPool
│                   Output: (36,)
│
├── Dense(16) + ReLU
├── Dropout(0.3)
└── Dense(N) + Softmax   — N = num_classes
```

**Key properties:**
- 15,851 parameters total (11-class) | ~15.0 KB weights | ~28.7 KB total flash
- ~968K MACs per inference (mel frontend alone: 515K MACs, 53%)
- BN layers fused into conv weights at NNoM export — zero BN overhead at runtime
- Mel frontend filters are learned end-to-end, not fixed

---

## Results

### 11-Class Model (down/go/left/no/off/on/right/stop/up/yes/unknown)

| Model | GSCD Float | GSCD int8 | Pi int8 | Params | Flash |
|---|---|---|---|---|---|
| mel_compact_4blk_ch36 (base) | 88.84% | 88.52% | 90.97% | 15,851 | 28.7 KB |
| mel_compact_4blk_ch36 (pi_mix_norm) | 89.71% | — | 90.97% | 15,851 | 28.7 KB |

`pi_mix_norm`: fine-tuned by mixing in Pi recordings during training with int8 peak-norm
augmentation. Same weight footprint as base, measurably better on real hardware.

### 4-Command Model (go / left / right / stop)

Fine-tuned from the 11-class backbone. Head replaced: Dense(11)→Dense(5, go/left/right/stop/unknown).
Two-phase training: freeze backbone 20 epochs LR=1e-3, then unfreeze all with cosine schedule.

| Model | GSCD Float | GSCD int8 | Pi int8 | Barbary int8 |
|---|---|---|---|---|
| mel_compact_4cmd (go/left/right/stop) | 95.56% | — | 95.16% | 96.88% |

Per-class Pi accuracy: go 95.5% | left 95.5% | right 96.8% | stop 92.9%

### Multi-Command Models (newds — best dataset, 11-class backbone fine-tuned)

Results from `results/multi_cmd_results_newds/`. Each row is a separate model fine-tuned for that
command pair/group, with unknown as catch-all.

| Commands | GSCD | Barbary | Pi |
|---|---|---|---|
| yes / no | 96.04% | 98.48% | — |
| left / right | 95.87% | 97.73% | — |
| go / stop | 95.41% | 96.97% | — |
| on / off | 94.99% | 97.73% | — |
| go / left / right / stop | 94.03% | 96.21% | — |
| up / down / left / right | 93.55% | 95.83% | — |
| on / off / up / down | 92.09% | 92.05% | — |
| up / down | 94.74% | — | 94.74% Pi |

---

## Training Methodology

### Int8 Peak-Norm Pipeline
All audio is normalized as: `scale = min(96 / peak, 64)` then cast to int8.
This matches exactly what the firmware does at inference time — training and deployment see the same
numeric range. Do not use int16 normalization or float normalization.

### PTQ Export (NNoM)
- Calibration method: KLD (Kullback-Leibler divergence)
- Quantization: per-channel, int8 weights + activations
- BN fusion: baked into preceding conv at export, no runtime BN
- Calibration data: 256 random uniform samples (sufficient for KLD to find scale factors)
- Script: `training/nnom_stat.py` → outputs `weights_4cmd.h` or equivalent

### Two-Phase Fine-Tuning (4-command)
1. Freeze all backbone layers, train new Dense(5) head: 20 epochs, LR=1e-3, Adam
2. Unfreeze all layers, cosine LR decay 1e-4→1e-6, patience=40 early stop on val_acc
3. Best checkpoint saved by ModelCheckpoint: `mel_compact_4cmd_int8norm.h5`

---

## File Index

### models/
| File | Description |
|---|---|
| `mel_compact_4blk_ch36_best.h5` | 11-class float baseline. Use as backbone for fine-tuning. |
| `mel_compact_4blk_ch36_int8norm.h5` | 11-class trained with int8 peak-norm. Best for NNoM export. |
| `mel_compact_4blk_ch36_pi_mix_norm_best.h5` | 11-class with Pi recordings mixed into training. Best Pi accuracy. |
| `mel_compact_4cmd_int8norm.h5` | 4-command (go/left/right/stop/unknown). Fine-tuned, int8norm. Deployed model. |

### training/
| File | Description |
|---|---|
| `train_improved.py` | Architecture definition. `build_mel_compact()` at line 100. Edit here for arch changes. |
| `train_v4.py` | Full 11-class training pipeline: data loading, augmentation, two-phase train, PTQ export. |
| `train_4cmd.py` | 4-command fine-tuning from backbone. Backbone path hardcoded at top. |
| `train_multi_cmd_newds.py` | Multi-command training loop. Pass `--cmds go stop` etc. Best-dataset variant. |
| `nnom_stat.py` | Standalone NNoM PTQ export. Load any `.h5`, run `generate_model()`, dump weights header. |

### weights_headers/
NNoM C array headers for firmware embedding. Include the relevant header and link against NNoM.

| File | For |
|---|---|
| `mel_compact_4blk_ch36_weights.h` | 11-class base model |
| `mel_compact_int8_peak_norm_weights.h` | 11-class int8 peak-norm variant |
| `weights_4cmd.h` | 4-command model. Has `KWS_NUM_CLASSES 5` and `KWS_CLASS_NAMES` macros. |

### results/
| Path | Description |
|---|---|
| `train_v4.log` | 11-class float training run log (88.84% final) |
| `train_pi_mix_norm.log` | pi_mix_norm fine-tuning log (89.71% GSCD) |
| `train_4cmd.log` | 4-command training log (converged epoch 94, 95.16% Pi) |
| `multi_cmd_results_newds/` | Per-command-set logs: GSCD, Barbary, Pi accuracy for 8 command groups |
| `build/kws_weights_int8norm_pi.log` | NNoM Pi evaluation: 90.97%, per-class breakdown |
| `build/kws_weights_no_qat_pi_mix_norm_pi.log` | pi_mix_norm NNoM Pi evaluation |
| `build/kws_weights_int8norm.log` | NNoM quantization stats (dec bits, ranges, per-layer) |

---

## How to Retrain

### 11-class base model
```bash
cd /workspace/Desktop/Models/Ziad/FinalModel
python3 train_v4.py 2>&1 | tee train_v4_new.log
```

### 4-command fine-tune (from 11-class backbone)
```bash
python3 train_4cmd.py 2>&1 | tee train_4cmd_new.log
# Edit BACKBONE_PATH at top of script to point to your backbone .h5
```

### Multi-command fine-tune
```bash
python3 train_multi_cmd_newds.py --cmds go left right stop 2>&1 | tee go_left_right_stop.log
```

### NNoM weights header export
```bash
python3 nnom_stat.py
# Edit model path and output path at top of script
# Output: /tmp/weights_stat_dump.h — copy to weights_headers/ and rename
```

---

## Firmware Deployment

The weights headers are consumed by the KWS-SoC firmware in `/workspace/Desktop/KWS-SoC/`.

```makefile
# In test/Makefile — 4-command target example:
$(CC) $(MEL_4CMD_CFLAGS) -o mel_compact_4cmd_xip_accel.elf ...
# CFLAGS include: -DKWS_WEIGHTS_HEADER='"weights_4cmd.h"'
#                 -DKWS_NUM_CLASSES is picked up from weights_4cmd.h
#                 -DKWS_CLASS_NAMES is picked up from weights_4cmd.h
```

The `weights_4cmd.h` header embeds `KWS_NUM_CLASSES` and `KWS_CLASS_NAMES` so `kws_bare_main.c`
adapts automatically — no firmware source edits needed when switching models.

For Spike RISC-V simulation:
```bash
cd /workspace/Desktop/kws-spike-validate
make run_mel_compact_4blk_ch36      # 11-class batch accuracy test
make build_mel_compact_4blk_ch36    # build only, no run
```
