import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

bias = True

@dataclass
class GPTConfig:
    vocab_size:  int = 30002
    n_embed:     int = 1536
    n_head:      int = 24
    n_layer:     int = 32
    dropout:   float = 0.0
    bias:       bool = True
    end_text_id: int = 0
    end_turn_id: int = 3

def sample_logits(logits, generated_tokens, rep_pen=1.0, temp=1.0, top_k=0, top_p=0.0, generator=None):

    for token_id in set(generated_tokens):
        score = logits[0][token_id]
        # If score is positive, decrease it. If negative, increase the magnitude
        if score < 0:
            logits[0][token_id] = score * rep_pen
        else:
            logits[0][token_id] = score / rep_pen

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
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.n_embed = n_embed
        self.dropout_p = dropout

        assert n_embed % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.head_size = n_embed // n_head

        self.qkv_projection = nn.Linear(n_embed, n_embed*3, bias=False)
        self.proj = nn.Linear(self.head_size * n_head, n_embed, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj.scale_init = True

        self.kv_cache = None
        self.current_seq_len = 0

        # torch.arange(0, D, 2) gets [0, 2, 4, ..., (D - 2)]
        inv_freq = 1.0 / (float(10000.0) ** (torch.arange(0, self.head_size, 2).float() / self.head_size))  # 1 / (base^(2i/D))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, ):
        B, T, C = x.shape
        qkv = self.qkv_projection(x)
        q, k, v = qkv.split(self.n_embed, dim=2)  # (B, T, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        q = q.to(torch.float32)
        k = k.to(torch.float32)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            inv_freq = self.inv_freq.to(torch.float32)

            if self.kv_cache is None or T != 1:
                t = torch.arange(T).to(inv_freq.device).to(torch.float32)
            else:
                t = torch.tensor([self.current_seq_len]).to(inv_freq.device).to(torch.float32)

            freqs = torch.einsum("i,j->ij", t, inv_freq) # [T, head_size/2]
            emb = torch.cat((freqs, freqs), dim=-1) # (T, head_size)
            
            cos = emb.cos().to(torch.float32).unsqueeze(0).unsqueeze(0)
            sin = emb.sin().to(torch.float32).unsqueeze(0).unsqueeze(0)
            #b, head, position, vect = 0, 16, 7, 8
            #print(f"K({k[b][head][position][vect]}, {k[b][head][position][(self.head_size//2)+vect]})")
            #print(f"Freq({inv_freq[vect]})")
            q = (q * cos) + (rotate_half(q) * sin)
            k = (k * cos) + (rotate_half(k) * sin)
            #print(f"K({k[b][head][position][vect]}, {k[b][head][position][(self.head_size//2)+vect]})")

        q = q.to(v.dtype)
        k = k.to(v.dtype)
    

        if self.kv_cache is None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True)
        else:
            self.kv_cache[0][:, :, self.current_seq_len : self.current_seq_len + T, :] = k
            self.kv_cache[1][:, :, self.current_seq_len : self.current_seq_len + T, :] = v

            self.current_seq_len += T

            cached_k = self.kv_cache[0][:, :, :self.current_seq_len, :]
            cached_v = self.kv_cache[1][:, :, :self.current_seq_len, :]

            out = F.scaled_dot_product_attention(q, cached_k, cached_v, attn_mask=None, dropout_p=0.0, is_causal=(T != 1))
        
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_embed)
        out = self.dropout(self.proj(out))
        return out
    
class FFWD(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4, bias=bias),
            nn.GELU(), 
            nn.Linear(n_embed * 4, n_embed, bias=bias), 
            nn.Dropout(dropout)
        )
        self.net[2].scale_init = True

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embed, n_head, dropout)
        self.ffwd = FFWD(n_embed, dropout)
        self.ln1 = nn.RMSNorm(n_embed)
        self.ln2 = nn.RMSNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.n_embed)
        self.blocks = nn.ModuleList([Block(self.config.n_embed, self.config.n_head, self.config.dropout) for _ in range(self.config.n_layer)])
        self.ln_f = nn.RMSNorm(self.config.n_embed)
        self.lm_head = nn.Linear(self.config.n_embed, self.config.vocab_size, bias=bias)

        #self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'scale_init'):
                std *= (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        x = self.token_embedding_table(index) # (B,T,C)
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets == None:
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
        shape = (index.shape[0], self.config.n_head, index.shape[-1] + max_new_tokens, self.config.n_embed // self.config.n_head)
        for block in self.blocks:
            block.sa.kv_cache = (torch.zeros(shape, device=device), torch.zeros(shape, device=device))
            block.sa.current_seq_len = 0

    def reset_cache(self):
        for block in self.blocks:
            block.sa.kv_cache = None
            block.sa.current_seq_len = 0