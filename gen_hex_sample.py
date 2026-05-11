import os
import numpy as np
from scipy.io import wavfile

# =========================================================
# CONFIGURATION
# =========================================================
FILE_PATH = "/workspace/Desktop/Main/FinalModel/Down_Male_Adult_Egypt_Open_Medium_1778000966594_16k_s16_monitor.wav"
OUTPUT_DIR = os.path.dirname(FILE_PATH)

HEX_OUT = os.path.join(OUTPUT_DIR, "down_audio_16k.hex")
BIN_OUT = os.path.join(OUTPUT_DIR, "down_audio_8k_spike.bin")

# Index for "down" based on: sorted(['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']) + ['unknown']
# ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', 'unknown']
DOWN_LABEL_IDX = 0 

# =========================================================
# WAV LOADER UTILITY
# =========================================================
def _load_wav(filepath, target_sr=8000, orig_sr=16000, dtype=np.float32):
    sr, samples = wavfile.read(filepath)

    import pdb; pdb.set_trace()
    if samples.dtype != np.int16:
        raise ValueError(f"Expected int16 audio, got {samples.dtype}")
         
    if sr != orig_sr:
        raise ValueError(f"Expected {orig_sr}, got {sr}")
    
    if samples.ndim > 1:
        samples = samples[:, 0]

    # PURE DOWNSAMPLING (DROP SAMPLES ONLY)
    factor = orig_sr // target_sr
    if factor > 1:
        samples = samples[::factor]

    # FORCE EXACT LENGTH (Truncate or Zero-pad)
    if len(samples) != target_sr:
        raise ValueError(f"Expected {target_sr} samples, got {len(samples)}")
    
    # FORMAT CONVERSION
    if dtype == np.float32:
        return (samples.astype(np.float32) / 32768.0)
    elif dtype == np.int8:
        return np.right_shift(samples, 8).astype(np.int8)
    
    return samples.astype(np.int16)

# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    print(f"Processing: {os.path.basename(FILE_PATH)}\n")

    # -----------------------------------------------------
    # TASK 1: 16kHz INT16 -> .HEX
    # -----------------------------------------------------
    print("Loading as 16kHz INT16...")
    audio_16k_int16 = _load_wav(
        FILE_PATH, 
        target_sr=16000, 
        orig_sr=16000, 
        dtype=np.int16
    )

    with open(HEX_OUT, "w") as f:
        for val in audio_16k_int16:
            # Mask to 16-bit to cleanly handle negative numbers in hex (Two's complement)
            f.write(f"{val & 0xFFFF:04X}\n")
            
    print(f" -> Saved HEX to: {HEX_OUT}")

    # -----------------------------------------------------
    # TASK 2: 8kHz INT8 -> Spike .BIN Format
    # -----------------------------------------------------
    print("\nLoading as 8kHz INT8...")
    audio_8k_int8 = _load_wav(
        FILE_PATH, 
        target_sr=8000, 
        orig_sr=16000, 
        dtype=np.int8
    )

    # Pack into Spike format: 1 int32 (n_clips), 128 uint8 (labels), 128 * 8000 int8 (audio)
    with open(BIN_OUT, "wb") as f:
        # 1. Write n_clips = 1
        f.write(np.array([1], dtype="<i4").tobytes())
        
        # 2. Write 128 labels (First is DOWN_LABEL_IDX, rest are 0 padded)
        labels_block = np.zeros(128, dtype=np.uint8)
        labels_block[0] = DOWN_LABEL_IDX
        f.write(labels_block.tobytes())
        
        # 3. Write 128 * 8000 audio samples (First slot gets our audio, rest are 0 padded)
        audio_block = np.zeros((128, 8000), dtype=np.int8)
        audio_block[0] = audio_8k_int8
        f.write(audio_block.tobytes())

    print(f" -> Saved Spike BIN to: {BIN_OUT}")
    print("\nDone!")