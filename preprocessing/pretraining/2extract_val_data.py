import numpy as np

# extracts a certain number of val tokens out of a pretokenized dset

# --- Configuration ---
input_file = '/home/user/Desktop/ml/gpt_dt/datasets/fineweb_t2/shards/s7.bin'        
new_train_file = '/home/user/Desktop/ml/gpt_dt/datasets/fineweb_t2/shards/s7_new.bin' 
val_file = '/home/user/Desktop/ml/gpt_dt/datasets/fineweb_t2/shards/s7_val.bin'         

val_tokens_target = 10_000_000         
dtype = np.uint16  # Change to np.uint32 if using Llama/modern tokenizers

# IMPORTANT: Set this to your tokenizer's End-Of-Text token ID
eot_id = 0
# ---------------------

print(f"Mapping {input_file}...")
full_data = np.memmap(input_file, dtype=dtype, mode='r')
total_tokens = len(full_data)

if total_tokens < val_tokens_target:
    raise ValueError("Dataset is smaller than the requested validation size!")

# 1. Find the target cut index
target_split = total_tokens - val_tokens_target

# 2. Scan forward to find the nearest EOT token
print(f"Target split at {target_split:,}. Scanning for nearest EOT token...")

# We load a chunk of tokens to search through (1 million should be plenty to find an EOT)
search_chunk = full_data[target_split - 1_000_000 : target_split]
eot_indices = np.where(search_chunk == eot_id)[0]

if len(eot_indices) == 0:
    raise ValueError("Could not find an EOT token near the target split! Check your eot_id.")
    
# We cut immediately AFTER the first EOT token we find
# This means the training set will end cleanly on an EOT, 
# and the val set will start cleanly on a new document.
exact_split_index = target_split - 1_000_000 + eot_indices[-1] + 1

actual_val_tokens = total_tokens - exact_split_index
print(f"Found EOT! Adjusting cut.")
print(f"Old val size: {val_tokens_target:,} -> New val size: {actual_val_tokens:,}")

# 3. Extract and save the validation set
print(f"\nWriting {actual_val_tokens:,} tokens to {val_file}...")
val_data = np.memmap(val_file, dtype=dtype, mode='w+', shape=(actual_val_tokens,))

chunk_size = 1024 * 1024 
for i in range(0, actual_val_tokens, chunk_size):
    end = min(i + chunk_size, actual_val_tokens)
    val_data[i:end] = full_data[exact_split_index + i : exact_split_index + end]
val_data.flush()

# 4. Extract and save the new training set
print(f"Writing {exact_split_index:,} tokens to {new_train_file}...")
train_data = np.memmap(new_train_file, dtype=dtype, mode='w+', shape=(exact_split_index,))

for i in range(0, exact_split_index, chunk_size):
    end = min(i + chunk_size, exact_split_index)
    train_data[i:end] = full_data[i:end]
train_data.flush()

print("\nDone! Clean document boundaries successfully maintained.")