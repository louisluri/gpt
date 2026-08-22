import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn import flash_attn_func, flash_attn_varlen_func

def sample_logits(logits, generated_tokens, rep_pen=1.0, temp=1.0, top_k=0, top_p=0.0, generator=None):

    for token_id in set(generated_tokens):
        # If score is positive, decrease it. If negative, increase the magnitude
        if logits[0][token_id] < 0:
            logits[0][token_id] *= rep_pen
        else:
            logits[0][token_id] /= rep_pen

    logits /= temp

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift indices to keep the first token that exceeds top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[..., indices_to_remove] = float('-inf')

    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_token

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_group = config.n_group
        self.n_embed = config.n_embed
        self.dropout_p = config.dropout

        assert config.n_embed % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        assert config.n_head % config.n_group == 0, "Number of heads must be divisible by number of groups for GQA"
        self.head_size = self.n_embed // self.n_head
        self.group_size = self.n_head // self.n_group
        self.kv_size = self.n_group * self.head_size

        self.qkv_projection = nn.Linear(self.n_embed, self.n_embed + (self.n_group * self.head_size * 2), bias=False)
        self.proj = nn.Linear(self.head_size * config.n_head, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.proj.scale_init = True

        self.kv_cache = None
        self.current_seq_len = 0

        # torch.arange(0, D, 2) gets [0, 2, 4, ..., (D - 2)]
        inv_freq = 1.0 / (float(10000.0) ** (torch.arange(0, self.head_size, 2).float() / self.head_size))  # 1 / (base^(2i/D))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, varlen_args, position_ids):
        # if varlen args are passed it will use the varlen_func, 
        # if the kv_caches are initialized before running, kv_caches will be used
        B, T, C = x.shape
        qkv = self.qkv_projection(x)

        q = qkv[..., : self.n_embed]
        k = qkv[..., self.n_embed : self.n_embed + self.kv_size] # (B, T, size)
        v = qkv[..., self.n_embed + self.kv_size :]

        q = q.view(B, T, self.n_head, self.head_size).to(torch.float32) # (B * T, n_head, head_size)
        k = k.view(B, T, self.n_group, self.head_size).to(torch.float32)
        v = v.view(B, T, self.n_group, self.head_size)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            inv_freq = self.inv_freq.to(torch.float32)
            
            freqs = torch.einsum("ij,k->ijk", position_ids, inv_freq) # [B, T, head_size/2]
            emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(2) # (B, T, 1, head_size)
            
            cos = emb.cos().to(torch.float32)
            sin = emb.sin().to(torch.float32)
            #position, head, vect = [10, 10, 14, 19,  3,  4,  7,  9], [ 5,  8, 12, 15,  2,  3,  4,  6], [10, 10,  4, 60, 53,  4, 17,  0]
            #print("before")
            #for i in range(len(position)):
            #    print(f"pos:{position[i]:02} head: {head[i]:02} item: {vect[i]:02} K({k_flat[position[i]][head[i]][vect[i]]}, {k_flat[position[i]][head[i]][(self.head_size//2)+vect[i]]}), Freq({inv_freq[vect[i]]})")
            
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
            #print("after")
            #for i in range(len(position)):
            #    print(f"pos:{position[i]:02} head: {head[i]:02} item: {vect[i]:02} K({k_flat[position[i]][head[i]][vect[i]]}, {k_flat[position[i]][head[i]][(self.head_size//2)+vect[i]]})")

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        # flash_attn handles the gqa kv broadcasting automatically
        if varlen_args:
            cu_seqlens, max_seqlen, _ = varlen_args
            q_flat = q.view(-1, self.n_head, self.head_size)
            k_flat = k.view(-1, self.n_group, self.head_size)
            v_flat = v.view(-1, self.n_group, self.head_size)
                    
            # Call the Varlen Kernel (Ensure q,k,v are in bfloat16!)
            out = flash_attn_varlen_func(q_flat, k_flat, v_flat, 
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen, 
                dropout_p=self.dropout_p if self.training else 0.0, causal=True)
        elif self.kv_cache is None:
            out = flash_attn_func(q, k, v, dropout_p=self.dropout_p if self.training else 0.0, causal=True)
        else:
            self.kv_cache[0][:, self.current_seq_len : self.current_seq_len + T] = k
            self.kv_cache[1][:, self.current_seq_len : self.current_seq_len + T] = v

            self.current_seq_len += T

            cached_k = self.kv_cache[0][:, :self.current_seq_len]
            cached_v = self.kv_cache[1][:, :self.current_seq_len]

            out = flash_attn_func(q, cached_k, cached_v, dropout_p=0.0, causal=(T != 1))
            
        # Un-flatten back to standard (B, T, C) for the rest of the network
        out = out.view(B, T, self.n_embed)
        out = self.dropout(self.proj(out))
        return out
        
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.n_embed, config.n_hidden, bias=config.bias)
        self.w2 = nn.Linear(config.n_embed, config.n_hidden, bias=config.bias)
        self.w3 = nn.Linear(config.n_hidden, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.w3.scale_init = True

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(x))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.swi = SwiGLU(config)
        self.ln1 = nn.RMSNorm(config.n_embed)
        self.ln2 = nn.RMSNorm(config.n_embed)

    def forward(self, x, varlen_args, position_ids):
        x = x + self.sa(self.ln1(x), varlen_args, position_ids)
        x = x + self.swi(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embed)
        self.blocks = nn.ModuleList([Block(config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.RMSNorm(self.config.n_embed)
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias=config.bias)

        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embed))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            if hasattr(module, 'scale_init'):
                nn.init.normal_(module.weight, mean=0.0, std=1 / math.sqrt(self.config.n_embed)  /  self.config.n_layer)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embed))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, index, targets=None, varlen_args=None):
        # if varlen args are passed it will use the varlen_func, they should be in the form (cu_seqlens, max_seqlen, position_ids)
        # if the kv_caches are initialized before running, kv_caches will be used
        B, T = index.shape

        if varlen_args is None:
            position_ids = torch.arange(T, device=index.device).to(torch.float32).unsqueeze(0)
        else:
            position_ids = varlen_args[2]

        x = self.token_embedding_table(index) # (B,T,C)
        for block in self.blocks:
            x = block(x, varlen_args, position_ids)
        
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits.view(B, T, -1), loss

    def generate(self, index, max_new_tokens, rep_pen=1.0, temp=1.0, top_k=0, top_p=0.0, seed=None):
        assert index.shape[0] == 1, "Only batch size of 1 is supported for inference"
        gen = None
        if seed is not None:
            gen = torch.Generator(device=index.device)
            gen.manual_seed(seed)

        #generated_tokens = []
        #with torch.no_grad():
        #    self.eval()
        #    torch.cuda.synchronize() # Wait for any previous operations to finish
        #    t0 = time.perf_counter()
        #    for _ in range(max_new_tokens):
        #        # get the logits for each char in the index
        #        logits, loss = self(index)
        #        # only look at the logits for each last letter
        #        logits = logits[:, -1, :]
        #        # choose one sample from our logits
        #        index_next = sample_logits(logits, generated_tokens, rep_pen, temp, top_k, top_p, gen)
        #        generated_tokens.append(index_next.item())
        #        #concatenate the letter choice for each batch onto the end of the existing char list
        #        index = torch.cat((index, index_next), dim=1) #(B, T+1)
        #        # stop generating if end of text token is generated
        #        if index_next.item() in [self.config.end_text_id, self.config.end_turn_id]:
        #            break
        #    torch.cuda.synchronize()
        #    t1 = time.perf_counter()
        #    self.train()
        #decoding_time = t1 - t0
        #tokens_per_second = (len(generated_tokens) - 1) / decoding_time if decoding_time > 0 else 0
        #print(f"Decoding Time: {decoding_time:.4f} seconds")
        #print(f"Generation Speed: {tokens_per_second:.2f} tokens/sec")
        #return generated_tokens
            
        self.build_kv_caches(index, max_new_tokens)

        generated_tokens = []
        with torch.no_grad():
            self.eval()

            torch.cuda.synchronize() # Wait for any previous operations to finish
            t0 = time.perf_counter()

            # get the logits for each char in the index
            logits, _ = self(index)
            # only look at the logits for each last letter
            logits = logits[:, -1, :]
            # choose one sample from our logits
            index_next = sample_logits(logits, generated_tokens, rep_pen, temp, top_k, top_p, gen)
            # append the next token to our generated tokens list
            generated_tokens.append(index_next.item())

            torch.cuda.synchronize() # Wait for prefill to finish
            t1 = time.perf_counter()
            
            for _ in range(max_new_tokens-1):
                # stop generating if end of text token is generated
                if index_next.item() in [self.config.end_text_id, self.config.end_turn_id]:
                    break

                logits, _ = self(index_next)
                logits = logits[:, -1, :]
                index_next = sample_logits(logits, generated_tokens, rep_pen, temp, top_k, top_p, gen)
                generated_tokens.append(index_next.item())

            torch.cuda.synchronize() # Wait for all decoding to finish
            t2 = time.perf_counter()

            self.train()
        self.reset_cache()

        prefill_time = t1 - t0
        decoding_time = t2 - t1
        tokens_per_second = (len(generated_tokens) - 1) / decoding_time if decoding_time > 0 else 0

        print(f"Prefill Time (TTFT): {prefill_time:.4f} seconds")
        print(f"Decoding Time: {decoding_time:.4f} seconds")
        print(f"Generation Speed: {tokens_per_second:.2f} tokens/sec")

        return generated_tokens
    
    def build_kv_caches(self, index, max_new_tokens):
        device = index.device
        shape = (index.shape[0], index.shape[-1] + max_new_tokens, self.config.n_group, self.config.n_embed // self.config.n_head)
        for block in self.blocks:
            block.sa.kv_cache = (torch.zeros(shape, device=device), torch.zeros(shape, device=device))
            block.sa.current_seq_len = 0

    def reset_cache(self):
        for block in self.blocks:
            block.sa.kv_cache = None
            block.sa.current_seq_len = 0