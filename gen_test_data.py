"""
Generate test_data.bin for Spike using raw INT8 parsing.

Format (little-endian, matches kws_nnom_main.c reader):
  int32  n_clips
  repeated:
    uint8  labels[128]
    int8   audio[128 * 8000]      (zero-padded if last batch is partial)
"""

import os
import numpy as np

# Import the unified loader from the previous file
# (Ensure the loading script is named load_data.py and in the same directory)
from load_data import load_speech_commands

# =========================================================
# CONFIGURATION
# =========================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/workspace/Desktop/Main/peaknorm"

CORE_COMMANDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
UNKNOWN_LABEL = 'unknown'
CLASSES = sorted(CORE_COMMANDS) + [UNKNOWN_LABEL]
NUM_CLASSES = len(CLASSES)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASSES)}

TARGET_SAMPLE_RATE = 8000
ORIG_SAMPLE_RATE = 16000
LABEL_BATCH = 128
SAMPLES = 8000

# =========================================================
# BINARY GENERATION SCRIPT
# =========================================================
if __name__ == "__main__":
    
    # 1. Load Dataset directly as INT8
    # We only care about the test split, so we ignore train and val
    _, _, test_data = load_speech_commands(
        DATA_DIR,
        CORE_COMMANDS,
        NUM_CLASSES,
        LABEL_TO_IDX,
        TARGET_SAMPLE_RATE,
        ORIG_SAMPLE_RATE,
        UNKNOWN_LABEL,
        dtype=np.int8, # <--- Requests INT8 directly from your loader
        max_unknown_per_split=None
    )

    x_test, y_test = test_data

    # 2. Squeeze trailing channel axis (n, 8000, 1) -> (n, 8000)
    x_int8 = x_test.reshape(x_test.shape[0], -1)
    
    # Cast labels to uint8 for the binary format
    y_uint8 = y_test.astype(np.uint8)

    n = x_int8.shape[0]
    out_path = os.path.join(HERE, 'test_data', 'peak_pi_test_data_int8.bin')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 3. Pack into Little-Endian Binary
    with open(out_path, 'wb') as f:
        # Write total clip count (int32)
        f.write(np.array([n], dtype="<i4").tobytes())
        
        for batch_start in range(0, n, LABEL_BATCH):
            batch_end = min(batch_start + LABEL_BATCH, n)
            
            # Write 128 uint8 labels
            labels_block = np.zeros(LABEL_BATCH, dtype=np.uint8)
            labels_block[:batch_end - batch_start] = y_uint8[batch_start:batch_end]
            f.write(labels_block.tobytes())
            
            # Write 128 * 8000 int8 audio (zero-padded automatically by np.zeros)
            audio_block = np.zeros((LABEL_BATCH, SAMPLES), dtype=np.int8)
            audio_block[:batch_end - batch_start] = x_int8[batch_start:batch_end]
            f.write(audio_block.tobytes())

    # 4. Summary Info
    print(f'\n--- Binary Generation Complete ---')
    print(f'Wrote to: {out_path}')
    print(f'Total clips: {n}')
    print(f'File size: {os.path.getsize(out_path)} bytes')
    print(f'Class distribution in test set:')
    for i, c in enumerate(CLASSES):
        print(f'  {i:2d} {c:>8s}: {int(np.sum(y_test == i))}')