import os
import time
import warnings
import contextlib
import shutil
import wandb
import torch._dynamo
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torchao.optim import AdamW8bit
from dataclasses import dataclass
from helpers import (
    get_optimizer_groups, 
    get_dataloader, 
    estimate_loss, 
    load_checkpoint, 
    save_checkpoint, 
    save_full_model, 
    get_varlen_args, 
    audit_local_parameters,
    get_lr, 
    set_seed
    )
from model import GPT, GPTConfig

torch._dynamo.config.optimize_ddp = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#---------------------------------------------------------------------------------------------
# CHECK ALL PARAMS BETWEEN THESE LINES ARE CORRECT BEFORE TRAINING
# MODEL HYPERPARAMS ARE AT THE TOP OF "model.py"

data_folder = "datasets/fineweb_t2"
block_size = 4096
map_step_size = block_size
train_dataset_path = f"{data_folder}/shards/s0.bin"
val_dataset_path = f"{data_folder}/shards/val.bin"
train_map_path = f"{data_folder}/{block_size}b_{map_step_size}s_maps/s0_map0.bin"
val_map_path = f"{data_folder}/{block_size}b_{map_step_size}s_maps/val_map0.bin"

#data_folder = "datasets/ift_t1"
#block_size = 1024
#map_step_size = block_size
#train_dataset_path = f"{data_folder}/packed_train_1024.bin"
#val_dataset_path = f"{data_folder}/packed_test_1024.bin"
#train_map_path = f"{data_folder}/packed_train_map0-{block_size}block-{map_step_size}step.bin"
#val_map_path = f"{data_folder}/packed_test_map0-{block_size}block-{map_step_size}step.bin"

load = 0 # load model from checkpoint?
save = 0  # save model?
log = 0  # log run with wandb?
load_model_number = "0" # the model number signifies which model it is
save_model_number = "0"
resume_step = "5"
abreviation = "test" # abreviation should be unique to the dataset and training session
load_path = f'checkpoints/m{load_model_number}/model{load_model_number}_{abreviation}{resume_step}'
save_folder = f'checkpoints/m{save_model_number}'
save_path = f'{save_folder}/model{save_model_number}_{abreviation}'
log_file = f'loss_curves/model{save_model_number}_{abreviation}_data.csv'

use_fsdp = 1  # if model too big for one gpu toggle fsdp on to use multiple gpus, if false will use DDP if multiple gpus are available
compile = 0 # use PyTorch 2.0 to compile the model to be faster

batch_size = 2   # effective batch size is batch_size * accumulation_steps * world_size
accumulation_steps = 128

save_steps = 5  # how many optimization steps between eval and saving model and data
log_steps = 5
assert save_steps % log_steps == 0, "save_steps must be a multiple of log_steps for consistent logging and checkpointing"

# special tokens
end_text_id = 0
system_id = 1
user_id = 2
assistant_id = 3
end_turn_id = 4
pad_id = 5

# adamw optimizer
learning_rate = 4e-4 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# learning rate decay settings
decay_lr = True
lr_decay_iters = 500 # total steps
warmup_iters = 30 # how many steps to warm up for
min_lr = 4e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

optim_step = 0
data_pos = 0

# Distributed settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda:1' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = torch.bfloat16 # use bfloat16 if amphere archetecture or newer (30 series or newer), else use float16 or float32

# END HYPERPARAMS
#----------------------------------------------------------------------------------------------------
distributed = int(os.environ.get("WORLD_SIZE", -1)) > 1  # is this a distributed run?
if distributed:
    dist.init_process_group(backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    master_process = local_rank == 0 # this process will do logging, checkpointing etc.
    ddp = not use_fsdp
    fsdp = use_fsdp
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    local_rank = 0
    world_size = 1
    device = torch.device(device)
    torch.cuda.set_device(device)
    ddp = False
    fsdp = False

tokens_per_step = accumulation_steps * world_size * batch_size * block_size
if master_process:
    print(f"\nTokens per step will be: {tokens_per_step:,}")
    print("\n--INITIALIZE/LOAD--")
if distributed: dist.barrier(device_ids=[local_rank])

#-----------------------------------------------------------------------------------------------------------
set_seed()
if fsdp:
    m = GPT(GPTConfig())
else:
    m = GPT(GPTConfig()).to(device)

if master_process: 
    print(f"\nTotal parameters {sum(p.numel() for p in m.parameters()):,}")

if ddp:
    m = DDP(m, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
elif fsdp:
    mp_policy = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=dtype)
    for block in m.blocks:
        fully_shard(block, mp_policy=mp_policy)
    fully_shard(m, mp_policy=mp_policy)

optim_groups = get_optimizer_groups(m, weight_decay)
optim = AdamW8bit(optim_groups, lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

if load:
    if fsdp:
        path = load_path
    else:
        path = f"{load_path}.pt"

    if os.path.exists(path):
        optim_step, data_pos = load_checkpoint(path, m, optim, fsdp=fsdp, local_rank=local_rank)
        print(f"Rank {local_rank} loaded succesfuly from (step: {optim_step}, data_pos: {data_pos})")
    else:
        print(f"No checkpoint found at {path}.")

if compile:
    m = torch.compile(m)

if fsdp:
    param_count = audit_local_parameters(m, local_rank)
else:
    param_count = sum(p.numel() for p in m.parameters())
    print(f"Rank {local_rank} total parameters: {param_count:,}")

lr = get_lr(optim_step, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
for param_group in optim.param_groups:
    param_group['lr'].fill_(lr)

if master_process: 
    print(f"Learning rate: {optim.param_groups[0]['lr']:.8e}")


@dataclass
class DataConfig:
    batch_size:    int = batch_size
    block_size:    int = block_size
    map_step_size: int = map_step_size
    pad_id:        int = pad_id
    assistant_id:  int = assistant_id
    end_turn_id:   int = end_turn_id
    num_replicas:  int = world_size
    rank:          int = local_rank

train_loader = get_dataloader(train_dataset_path, train_map_path, DataConfig(), start_pos=data_pos)
val_loader = get_dataloader(val_dataset_path, val_map_path, DataConfig())


if master_process: print("\n--TRAINING LOOP--")

if distributed: dist.barrier(device_ids=[local_rank])

#-----------------------------------------------------------------------------------------------------------
try:
    if master_process: print(f"{time.localtime().tm_hour:02}:{time.localtime().tm_min:02}:{time.localtime().tm_sec:02}")
    val_loss = estimate_loss(m, val_loader, end_text_id, device, dtype, world_size)   # estimate a base loss before training session
    if master_process: 
        print(f"\n(step: {optim_step}, data_pos: {data_pos}), Val Loss: {val_loss:.4f}")
        print(f"{time.localtime().tm_hour:02}:{time.localtime().tm_min:02}:{time.localtime().tm_sec:02}")

    if log and master_process:
        run = wandb.init(
            entity="laluriinc", 
            project="gpt-dt", 
            name=f"model{save_model_number}_{abreviation}", 
            settings=wandb.Settings(console="off"),
            config={
                "model": {
                    "param_count": param_count,
                    "vocab_size": GPTConfig.vocab_size,
                    "n_embed": GPTConfig.n_embed,
                    "hidden_dim": GPTConfig.n_hidden,
                    "n_head": GPTConfig.n_head,
                    "n_group": GPTConfig.n_group, 
                    "n_layer": GPTConfig.n_layer,
                    "dropout": GPTConfig.dropout,
                    "bias": GPTConfig.bias
                    },
                "optim": {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "beta1": beta1,
                    "beta2": beta2,
                    "decay_lr": decay_lr,
                    "warmup_iters": warmup_iters,
                    "lr_decay_iters": lr_decay_iters,
                    "min_lr": min_lr,
                    "grad_clip": grad_clip
                    },
                "train": {
                    "batch_size": batch_size,
                    "accumulation_steps": accumulation_steps,
                    "world_size": world_size,
                    "block_size": block_size,
                    "tokens_per_step": tokens_per_step
                    },
                "system": {
                    "fsdp": fsdp,
                    "ddp": ddp,
                    "dtype": str(dtype)
                    },
                "tokens": {
                    "end_text_id": end_text_id,
                    "system_id": system_id,
                    "user_id": user_id,
                    "assistant_id": assistant_id,
                    "end_turn_id": end_turn_id,
                    "pad_id": pad_id
                    }
                }
            )

        wandb.run.log_code(root=".", 
            include_fn=lambda path: path.endswith(".py") and os.path.dirname(os.path.abspath(path)) == os.path.abspath("."))

        wandb.log({"val_loss": val_loss, "learning_rate": optim.param_groups[0]['lr']}, step=optim_step)

    if distributed: dist.barrier(device_ids=[local_rank])

    train_start_event = torch.cuda.Event(enable_timing=True)
    train_end_event = torch.cuda.Event(enable_timing=True)
    
    if save:
        os.makedirs(save_folder, exist_ok=True)

    optim.zero_grad(set_to_none=True)
    running_loss = torch.tensor(0.0, device=device)
    max_norm_window = torch.tensor(0.0, device=device)
    train_start_event.record()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        varlen_args = get_varlen_args(x, end_text_id)

        # Determine if we need to block synchronization for gradient accumulation
        is_accumulating = (batch_idx + 1) % accumulation_steps != 0
        if fsdp:
            m.set_requires_gradient_sync(not is_accumulating)
            sync_context = contextlib.nullcontext()
        else:
            sync_context = m.no_sync() if (is_accumulating and ddp) else contextlib.nullcontext()

        with sync_context:
            with torch.amp.autocast('cuda', dtype=dtype):
                logits, loss = m(x, y, varlen_args)
                loss /= accumulation_steps
            loss.backward()

        running_loss += loss.detach()

        if not is_accumulating:
            if grad_clip > 0.0:
                norm_tensor = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=grad_clip)
                max_norm_window = torch.max(max_norm_window, norm_tensor.detach())

            optim.step()
            optim.zero_grad(set_to_none=True)
            optim_step += 1

            lr = get_lr(optim_step, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
            for param_group in optim.param_groups:
                param_group['lr'].fill_(lr)

            if optim_step % log_steps == 0:
                if distributed:
                    dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
                    train_loss = (running_loss.item() / world_size) / log_steps
                else:
                    train_loss = running_loss.item() / log_steps

                if master_process:
                    if log:
                        wandb.log({
                            "train_loss": train_loss, 
                            "norm_peak": max_norm_window.item(), 
                            "learning_rate": optim.param_groups[0]['lr']
                            }, step=optim_step)
                    else:
                        print(f"Step {optim_step}, Train Loss: {train_loss:.4f}, norm_peak: {max_norm_window.item():.4f}")
                        print(f"{time.localtime().tm_hour:02}:{time.localtime().tm_min:02}:{time.localtime().tm_sec:02}")

                running_loss = torch.tensor(0.0, device=device)
                max_norm_window = torch.tensor(0.0, device=device)

            if optim_step % save_steps == 0:
                train_end_event.record() # as soon as the gpu is done
                torch.cuda.synchronize() # sync before reading
                # elapsed_time returns milliseconds, so divide by 1000.0
                train_time_sec = train_start_event.elapsed_time(train_end_event) / 1000.0

                tic = time.perf_counter()
                val_loss = estimate_loss(m, val_loader, end_text_id, device, dtype, world_size)
                torch.cuda.synchronize()
                eval_time_sec = time.perf_counter() - tic
                torch.cuda.empty_cache()
                
                data_pos += batch_size*world_size*accumulation_steps*save_steps

                if master_process: 
                    print(f"\n{time.localtime().tm_hour:02}:{time.localtime().tm_min:02}:{time.localtime().tm_sec:02}")
                    print(f"(step: {optim_step}, data_pos: {data_pos}), Val Loss: {val_loss:.4f}")
                    print(f"Learning rate: {optim.param_groups[0]['lr']:.8e}")

                if save:
                    save_checkpoint(optim_step, data_pos, m, optim, save_path, fsdp, local_rank, master_process)

                    if fsdp and optim_step % (save_steps * 5) == 0:
                        save_full_model(m, save_path, optim_step, local_rank)
                        
                    if master_process:
                        del_step = optim_step - (save_steps * 3)
                        del_checkpoint_path = f"{save_path}{del_step}"
                        if del_step % (save_steps * 20) != 0:
                            if os.path.exists(f"{del_checkpoint_path}.pt"):
                                os.remove(f"{del_checkpoint_path}.pt")
                                print(f"Deleted old checkpoint: {f'{del_checkpoint_path}.pt'}")
                            if os.path.exists(del_checkpoint_path):
                                shutil.rmtree(del_checkpoint_path)
                                print(f"Deleted old DCP folder: {del_checkpoint_path}")

                if log and master_process:
                    wandb.log({
                        "val_loss": val_loss,
                        "tokens_per_sec": tokens_per_step / (train_time_sec / save_steps),
                        "time_per_optim_step(min)": train_time_sec / save_steps / 60,
                        "time_eval(min)": eval_time_sec / 60
                        }, step=optim_step)
                elif master_process:
                    print(f"tps: {tokens_per_step / (train_time_sec / save_steps):.5f}")
                
                if distributed: dist.barrier(device_ids=[local_rank])
                train_start_event.record()
    print("Done!")
except Exception as e:
    print(f"\nError caught on rank {dist.get_rank()}: {e}")
    torch.cuda.empty_cache()
    os._exit(1)
finally:
    if save:
        save_checkpoint(optim_step, data_pos, m, optim, save_path, fsdp, local_rank, master_process)
        if fsdp: 
            save_full_model(m, save_path, optim_step, local_rank)

    if log and master_process: 
        run.finish()

    print("\nCleaning up GPU resources...")
    torch.cuda.empty_cache()

    if distributed: 
        destroy_process_group()