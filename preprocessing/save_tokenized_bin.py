from datasets import load_dataset
import numpy as np
import multiprocessing as mp
from tokenizers import Tokenizer
import gc
import pyarrow as pa
import ctypes

tokenizer_file = '/home/user/Desktop/ml/gpt_dt/tokenizer/tokenizer-03.json'

num_proc = 8 # 8 p cores

def process_shard(shard_idx):
    print(f"[Process {shard_idx}] Starting continuous stream...")
    
    tokenizer = Tokenizer.from_file(tokenizer_file)
    eot_id = tokenizer.token_to_id("<|end_of_text|>")
    
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)
    
    dataset = dataset.shard(num_shards=num_proc, index=shard_idx)

    filename = f"t3_fw_data_shard_{shard_idx:02d}.bin"
    tokens_written = 0
    last_print = 0
    
    with open(filename, 'wb') as f:
        
        for document in dataset:
            ids = tokenizer.encode(document["text"], add_special_tokens=False).ids
            ids.append(eot_id)
            
            arr = np.array(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            
            tokens_written += len(ids)

            if tokens_written - last_print >= 100_000_000:
                print(f"[Shard {shard_idx:02d}] {tokens_written / 1e9:.3f} Billion tokens processed.")
                last_print = tokens_written

                gc.collect()
                pa.default_memory_pool().release_unused()

                try:
                    ctypes.CDLL('libc.so.6').malloc_trim(0)
                except Exception:
                    print("except")
                    pass

    print(f"[Process {shard_idx}] Finished processing entire shard! Total tokens: {tokens_written:,}")

# --- EXECUTION ---
if __name__ == '__main__':
    print(f"Launching {num_proc} parallel workers...")
    mp.set_start_method('spawn', force=True)
    
    with mp.Pool(num_proc) as pool:
        pool.map(process_shard, range(num_proc))
        
    print("\nAll 100 Billion tokens processed successfully!")