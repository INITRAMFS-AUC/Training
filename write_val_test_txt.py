import os
import random

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = "/workspace/Desktop/Main/pi_dataset"
VAL_PERCENT  = 0.1  # 10% for validation
TEST_PERCENT = 0.1 # 10% for testing

VAL_FILE = "validation_list.txt"
TEST_FILE = "testing_list.txt"

# Set a random seed so your splits are reproducible if you run this multiple times
random.seed(42)

# =========================================================
# MAIN EXECUTION
# =========================================================
def generate_splits():
    if not os.path.exists(BASE_DIR):
        print(f"ERROR: Directory not found -> {BASE_DIR}")
        return

    # Get all subdirectories (classes) in the base directory
    classes = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    if not classes:
        print("No class directories found.")
        return

    val_lines = []
    test_lines = []
    total_processed = 0
    
    print(f"Scanning {BASE_DIR}...")
    
    for cls in sorted(classes):
        cls_dir = os.path.join(BASE_DIR, cls)
        
        # Get all files in this specific class folder
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))]
        
        if not files:
            continue
            
        # Shuffle files to ensure a random selection for the splits
        random.shuffle(files)
        
        total_files = len(files)
        total_processed += total_files
        
        # Calculate exactly how many files to allocate
        val_count = int(total_files * VAL_PERCENT)
        test_count = int(total_files * TEST_PERCENT)
        
        # Slice the shuffled list
        val_files = files[:val_count]
        test_files = files[val_count : val_count + test_count]
        
        # Format as "Class/filename" and add to lists
        for f in val_files:
            val_lines.append(f"{cls}/{f}\n")
            
        for f in test_files:
            test_lines.append(f"{cls}/{f}\n")
            
        print(f"  {cls}: {total_files} total -> {val_count} val | {test_count} test")

    # Write the lists to their respective text files
    with open(VAL_FILE, "w") as f:
        f.writelines(val_lines)
        
    with open(TEST_FILE, "w") as f:
        f.writelines(test_lines)

    # Print summary
    print("\n--- SUMMARY ---")
    print(f"Total files processed: {total_processed}")
    print(f"Saved {len(val_lines)} items to {VAL_FILE}")
    print(f"Saved {len(test_lines)} items to {TEST_FILE}")

if __name__ == "__main__":
    generate_splits()