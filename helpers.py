import gc
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, get_state_dict

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CustomSampler(Sampler):
    def __init__(self, map_path, start_pos=0, num_replicas=1, rank=0):
        indices_np = np.memmap(map_path, dtype=np.int64, mode='r')
        self.indices = torch.from_numpy(indices_np)

        self.start_pos = start_pos
        assert start_pos < len(self.indices), "start_pos is out of bounds"
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        remaining = self.indices[self.start_pos:]
        tot_samples = len(remaining)
        total_valid_samples = tot_samples - (tot_samples % self.num_replicas)  # trim to a multiple of num_replicas
        my_indices = remaining[self.rank:total_valid_samples:self.num_replicas]
        for idx in my_indices:
            yield idx.item()

    def __len__(self):
        return len(self.indices) # incorrect
        
class BinDataset(Dataset):
    def __init__(self, data_path, config):
        self.block_size = config.block_size
        self.map_step = config.map_step_size
        self.pad_id = config.pad_id
        self.assistant_id = config.assistant_id
        self.end_turn_id = config.end_turn_id

        # creates a pointer to the file without loading it.
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
    def __len__(self):
        return (len(self.data) - 1) // self.map_step

    def __getitem__(self, i):
        i = i * self.map_step
        chunk = self.data[i : i + self.block_size + 1]
        
        t = torch.from_numpy(chunk.astype(np.int64))
        
        x = t[:-1] # Input
        y = t[1:].clone()  # Target (shifted right)

        y[y == self.pad_id] = -100

        assistant_pos = (y == self.assistant_id).nonzero(as_tuple=True)[0]

        if len(assistant_pos) > 0:
            end_turn_pos = (y == self.end_turn_id).nonzero(as_tuple=True)[0]
            loss_mask = torch.zeros_like(y, dtype=torch.bool)

            for asst in assistant_pos:
                valid_ends = end_turn_pos[end_turn_pos > asst]
        
                if len(valid_ends) > 0:
                    # The very next end token is just the first one in the filtered list
                    next_end = valid_ends[0]
                else:
                    # The sequence hit your max_length and truncated the text 
                    # before the assistant finished! Just go to the end of the tensor.
                    next_end = y.size(0)
                    
                # asst + 2 ensures we don't train on the <|assistant|> token or the "\n"
                # next_end + 1 ensures we include the <|end_of_turn|> token in the loss
                loss_mask[asst + 2 : next_end + 1] = True                
            y[~loss_mask] = -100

        return x, y
    
def get_dataloader(data_path, map_path, config, start_pos=0):

    dataset = BinDataset(data_path, config)
    sampler = CustomSampler(map_path, start_pos, num_replicas=config.num_replicas, rank=config.rank)
    loader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)

    return loader

@torch.no_grad()
def estimate_loss(m, val_loader, end_text_id, device, dtype, world_size):
    m.eval()
    running_loss = torch.tensor(0.0, device=device)
    eval_iters = torch.tensor(0, device=device)
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        varlen_args = get_varlen_args(x, end_text_id)
        with torch.amp.autocast('cuda', dtype=dtype):
            _, loss = m(x, y, varlen_args)

        running_loss += loss
        eval_iters += 1

        if eval_iters >= 10: break

    if world_size > 1:
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_iters, op=dist.ReduceOp.SUM)

    final_loss = running_loss.item() / eval_iters.item()

    m.train()
    return final_loss

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    # linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    # if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def get_optimizer_groups(model, weight_decay):
    # Separate parameters into those that will experience weight decay and those that won't
    optim_groups = [
        {"params": [p for n, p in model.named_parameters() if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    return optim_groups

def get_inner_model(model):
    model_to_load = model
    while hasattr(model_to_load, 'module') or hasattr(model_to_load, '_orig_mod'):
        if hasattr(model_to_load, 'module'):
            model_to_load = model_to_load.module
        elif hasattr(model_to_load, '_orig_mod'):
            model_to_load = model_to_load._orig_mod
    return model_to_load

def clean_state_dict(state):
    clean_state = {}
    for key, value in state.items():
        new_key = key
        while True:
            if new_key.startswith("_orig_mod."):
                new_key = new_key.replace("_orig_mod.", "", 1)
            elif new_key.startswith("module."):
                new_key = new_key.replace("module.", "", 1)
            else:
                break # Key is clean
        clean_state[new_key] = value
    return clean_state

def save_checkpoint(step, data_pos, model, optimizer, path, fsdp, local_rank, master_process):
    """saves the current model and optimizer state to the checkpoint path specified at the top
    Prints to confirm completion"""
    if fsdp:
        m_sd, o_sd = get_state_dict(model, optimizer)
    else:
        m_sd = clean_state_dict(model.state_dict())
        o_sd = optimizer.state_dict()

    checkpoint = {
        'model': m_sd,
        'optimizer': o_sd,
        'meta': {'step': step, 'data_pos': data_pos}
        }
    
    if fsdp:
        dcp.state_dict_saver.save(checkpoint, checkpoint_id=f"{path}{step}")
        print(f"Rank {local_rank} checkpoint (step: {step}, data_pos: {data_pos}) Saved")
    elif master_process:
        torch.save(checkpoint, f"{path}{step}.pt")
        print(f"Checkpoint (step: {step}, data_pos: {data_pos}) Saved")

    del checkpoint, m_sd, o_sd
    gc.collect()
    torch.cuda.empty_cache()

def load_checkpoint(path, model, optimizer, fsdp=False, local_rank=0):
    print(f"Rank {local_rank} loading checkpoint from {path}")
    
    if fsdp:
        m_sd, o_sd = get_state_dict(model, optimizer)
        checkpoint = {
            'model': m_sd,
            'optimizer': o_sd,
            'meta': {'step': 0, 'data_pos': 0}
            }
        dcp.state_dict_loader.load(checkpoint, checkpoint_id=path)
        dist.barrier(device_ids=[local_rank])
    else:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        get_inner_model(model).load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    optim_step, data_pos = checkpoint['meta']['step'], checkpoint['meta']['data_pos']

    del checkpoint, m_sd, o_sd
    gc.collect()
    torch.cuda.empty_cache()

    return optim_step, data_pos

def save_full_model(model, path, step, local_rank):
    """for fsdp runs only, saves the full model unsharded"""
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state_dict = get_model_state_dict(model, options=options)
    
    if local_rank == 0:
        torch.save(full_state_dict, f"{path}{step}.pt")
        print(f"Full model saved to {path}{step}.pt")

    dist.barrier(device_ids=[local_rank])

def audit_local_parameters(model, rank):
    local_params = 0
    global_params = 0
    ignored_params = 0

    for p in model.parameters():
        global_params += p.numel()
        
        # If it's sharded by FSDP2, count the local piece
        if isinstance(p, DTensor):
            local_params += p.to_local().numel()
        # If it's an "Ignored Module", the full thing is local
        else:
            local_params += p.numel()
            ignored_params += p.numel()
            
    print(f"[Rank {rank}] Global Ledger: {global_params:,}")
    print(f"[Rank {rank}] Local Vault:   {local_params:,}")
    print(f"[Rank {rank}] Ignored/Full:  {ignored_params:,}")

    return global_params

def get_varlen_args(x, end_text_id):
    B, T = x.shape
    x_flat = x.reshape(-1)
    
    # Get cu_seqlens
    end_text_indices = (x_flat == end_text_id).nonzero(as_tuple=True)[0]
    doc_starts = (end_text_indices + 1).to(torch.int32)
    block_boundaries = torch.arange(0, (B * T) + 1, step=T, device=x.device, dtype=torch.int32)
    cu_seqlens = torch.cat([doc_starts, block_boundaries])
    cu_seqlens = torch.unique(cu_seqlens).to(torch.int32)

    seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = seq_lengths.max().item()
    
    # Build resetting position_ids!
    # Create a boolean mask of document starts
    is_start = torch.zeros_like(x_flat, dtype=torch.bool)
    is_start[cu_seqlens[:-1].long()] = True # Mark where new docs start (ignore the out-of-bounds last one)
    
    # Cumulative sum creates a unique ID for each document (0, 0, 0, 1, 1, 1...)
    doc_ids = torch.cumsum(is_start.int(), dim=0) - 1
    
    # Standard arange from 0 to B*T
    global_positions = torch.arange(x_flat.size(0), device=x.device)
    
    # We subtract the starting index of each document to reset the count to 0!
    # cu_seqlens[doc_ids] gives us an array of offsets: [0, 0, 0, 3, 3, 3...]
    position_ids = global_positions - cu_seqlens[doc_ids]

    # pad up to 32 so as not to cause compiler graph breaks
    current_len = cu_seqlens.shape[-1]
    if current_len < 32:
        pad_amount = 32 - current_len
        # Pad with your block_size (e.g., B*T)
        padding = torch.full((pad_amount,), B*T, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        cu_seqlens = torch.cat([cu_seqlens, padding])
    elif current_len > 32:
        print("***got batch with more than the expected maximum number of docs, this  will result in a recompilation***")
    
    return cu_seqlens, max_seqlen, position_ids.reshape(B, T)