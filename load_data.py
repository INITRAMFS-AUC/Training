import os
import numpy as np
from scipy.io import wavfile

# =========================================================
# SPLIT FILES
# =========================================================
def _read_split_list(filepath):
    if not os.path.exists(filepath):
        return set()

    with open(filepath, "r") as f:
        return set(line.strip() for line in f if line.strip())


# =========================================================
# WAV LOADER (SUPPORTS INT16, INT8, OR FLOAT32)
# =========================================================
def _load_wav(filepath, target_sr=8000, orig_sr=16000, dtype=np.float32):
    sr, samples = wavfile.read(filepath)

    if samples.dtype != np.int16:
        raise ValueError(f"Expected int16 audio, got {samples.dtype}")
         
    if sr != orig_sr:
        raise ValueError(f"Expected {orig_sr}, got {sr}")
    
    if samples.ndim > 1:
        samples = samples[:, 0]

    # -----------------------------------------------------
    # PURE DOWNSAMPLING (DROP SAMPLES ONLY)
    # -----------------------------------------------------
    factor = orig_sr // target_sr
    if factor > 1:
        samples = samples[::factor]

    # -----------------------------------------------------
    # Check Length
    # -----------------------------------------------------
    if len(samples) != target_sr:
        raise ValueError(f"Expected {target_sr} samples, got {len(samples)}")

    # -----------------------------------------------------
    # Format Conversion
    # -----------------------------------------------------
    if dtype == np.float32:
        # Scale int16 [-32768, 32767] to float32 [-1.0, 1.0]
        return (samples.astype(np.float32) / 32768.0)
    elif dtype == np.int8:
        # Scale int16 [-32768, 32767] to int8 [-128, 127]
        return np.right_shift(samples, 8).astype(np.int8)
    
    return samples.astype(np.int16)


# =========================================================
# BACKGROUND NOISE LOADER
# =========================================================
def _get_background_segments(data_dir, target_sr, orig_sr, dtype=np.float32):
    bg_dir = os.path.join(data_dir, "_background_noise_")
    segments = []

    if not os.path.exists(bg_dir):
        return segments

    for fname in sorted(os.listdir(bg_dir)):
        if not fname.endswith(".wav"):
            continue

        fpath = os.path.join(bg_dir, fname)
        
        try:
            sr, samples = wavfile.read(fpath)
        except Exception:
            continue
            
        if sr != orig_sr:
            continue
            
        if samples.ndim > 1:
            samples = samples[:, 0]

        # PURE DOWNSAMPLING
        factor = orig_sr // target_sr
        if factor > 1:
            samples = samples[::factor]

        # SPLIT INTO SEGMENTS
        for i in range(0, len(samples) - target_sr + 1, target_sr):
            seg = samples[i:i + target_sr]

            if len(seg) == target_sr:
                # Format Conversion
                if dtype == np.float32:
                    segments.append((seg.astype(np.float32) / 32768.0))
                elif dtype == np.int8:
                    segments.append(np.right_shift(seg, 8).astype(np.int8))
                elif dtype == np.int16:
                    segments.append(seg.astype(np.int16))
                else:
                    raise TypeError(f"Datatype {dtype} is not supported")

    return segments


# =========================================================
# MAIN LOADER
# =========================================================
def load_speech_commands(
    DATA_DIR,
    CORE_COMMANDS,
    NUM_CLASSES,
    LABEL_TO_IDX,
    TARGET_SAMPLE_RATE,
    ORIG_SAMPLE_RATE,
    UNKNOWN_LABEL,
    dtype,
    max_unknown_per_split=None
):
    print(f"Loading dataset from {DATA_DIR}")
    print(f"Core commands: {CORE_COMMANDS}")
    print(f"Target dtype: {dtype.__name__}")
    print(f"Num classes: {NUM_CLASSES}\n")

    test_files = _read_split_list(os.path.join(DATA_DIR, "testing_list.txt"))
    val_files  = _read_split_list(os.path.join(DATA_DIR, "validation_list.txt"))

    core_files = {"train": [], "val": [], "test": []}
    noncore_files = {"train": [], "val": [], "test": []}

    unknown_idx = LABEL_TO_IDX[UNKNOWN_LABEL]

    # -----------------------------------------------------
    # COLLECT FILES
    # -----------------------------------------------------
    for word in sorted(os.listdir(DATA_DIR)):
        word_dir = os.path.join(DATA_DIR, word)

        if not os.path.isdir(word_dir) or word.startswith("_"):
            continue

        is_core = word in CORE_COMMANDS
        label = LABEL_TO_IDX[word] if is_core else unknown_idx

        for fname in sorted(os.listdir(word_dir)):
            if not fname.endswith(".wav"):
                continue

            rel = f"{word}/{fname}"
            path = os.path.join(word_dir, fname)

            if rel in test_files:
                key = "test"
            elif rel in val_files:
                key = "val"
            else:
                key = "train"

            if is_core:
                core_files[key].append((path, label))
            else:
                noncore_files[key].append((path, label))

    # -----------------------------------------------------
    # BACKGROUND NOISE
    # -----------------------------------------------------
    rng = np.random.default_rng(42)

    bg_segments = _get_background_segments(
        DATA_DIR,
        TARGET_SAMPLE_RATE,
        ORIG_SAMPLE_RATE,
        dtype=dtype
    )

    rng.shuffle(bg_segments)

    n_bg = len(bg_segments)
    tr_end = int(0.8 * n_bg)
    vl_end = int(0.9 * n_bg)

    bg_pools = {
        "train": bg_segments[:tr_end],
        "val": bg_segments[tr_end:vl_end],
        "test": bg_segments[vl_end:]
    }

    # -----------------------------------------------------
    # Build Splits
    # -----------------------------------------------------
    results = {}

    for key in ["train", "val", "test"]:
        c_files = core_files[key]
        nc_files = noncore_files[key]

        core_counts = [lbl for _, lbl in c_files]

        if len(core_counts) > 0:
            _, counts = np.unique(core_counts, return_counts=True)
            target_unknown = int(np.mean(counts))
        else:
            target_unknown = 0

        if max_unknown_per_split is not None:
            target_unknown = min(target_unknown, max_unknown_per_split)

        req_nc = int(target_unknown * 0.7)
        req_bg = int(target_unknown * 0.2)

        act_nc = min(req_nc, len(nc_files))
        act_bg = min(req_bg, len(bg_pools[key]))
        act_sil = target_unknown - act_nc - act_bg

        sampled_nc = []
        if act_nc > 0:
            idx = rng.choice(len(nc_files), act_nc, replace=False)
            sampled_nc = [nc_files[i] for i in idx]

        audio_list = []
        labels_list = []

        # 1. CORE + UNKNOWN WORDS
        for fpath, lbl in c_files + sampled_nc:
            try:
                audio = _load_wav(fpath, TARGET_SAMPLE_RATE, ORIG_SAMPLE_RATE, dtype)
                audio_list.append(audio)
                labels_list.append(lbl)
            except ValueError:
                # Skip files that don't match exactly 1-second
                continue

        # 2. BACKGROUND NOISE
        if act_bg > 0:
            idx = rng.choice(len(bg_pools[key]), act_bg, replace=False)
            for i in idx:
                audio_list.append(bg_pools[key][i])
                labels_list.append(unknown_idx)

        # 3. GENERATE SILENCE
        for _ in range(act_sil):
            if dtype == np.float32:
                sil = rng.normal(0, 0.02, TARGET_SAMPLE_RATE).astype(np.float32)
                sil = np.clip(sil, -1.0, 1.0)
            elif dtype == np.int16:
                sil = rng.normal(0, 300, TARGET_SAMPLE_RATE)
                sil = np.clip(sil, -32768, 32767).astype(np.int16)
            elif dtype == np.int8:
                sil = rng.normal(0, 1.17, TARGET_SAMPLE_RATE)
                sil = np.clip(sil, -128, 127).astype(np.int8)

            audio_list.append(sil)
            labels_list.append(unknown_idx)

        x = np.array(audio_list, dtype=dtype)[..., np.newaxis]
        y = np.array(labels_list, dtype=np.int32)

        idx = rng.permutation(len(x))
        results[key] = (x[idx], y[idx])

    # -----------------------------------------------------
    # PRINT DISTRIBUTION
    # -----------------------------------------------------
    print("\nDataset Distribution:")
    for key in ["train", "val", "test"]:
        y = results[key][1]
        print(f"\n{key.upper()}:")
        for c, i in LABEL_TO_IDX.items():
            print(f"  {c}: {np.sum(y == i)}")

    return results["train"], results["val"], results["test"]

# =========================================================
# EXECUTION SCRIPT
# =========================================================
if __name__ == "__main__":
    DATA_DIR = "/workspace/Desktop/Main/dataset"
    CORE_COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    UNKNOWN_LABEL = 'unknown'
    CLASSES = sorted(CORE_COMMANDS) + [UNKNOWN_LABEL]
    NUM_CLASSES = len(CLASSES)
    LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}
    
    TARGET_SAMPLE_RATE = 8000
    ORIG_SAMPLE_RATE = 16000

    # Example: Loading as int16
    train, val, test = load_speech_commands(
        DATA_DIR,
        CORE_COMMANDS,
        NUM_CLASSES,
        LABEL_TO_IDX,
        TARGET_SAMPLE_RATE,
        ORIG_SAMPLE_RATE,
        UNKNOWN_LABEL,
        dtype=np.int16, # <--- TOGGLE THIS: np.int16, np.int8, or np.float32
        max_unknown_per_split=None
    )

    x_train, y_train = train
    print(f"\nFinished! Train data shape: {x_train.shape}, dtype: {x_train.dtype}")