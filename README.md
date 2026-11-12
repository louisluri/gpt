# Local Hardware GPT Training

This code trains a Generative Pretrained Transformer on your local computer with single or multiple gpu setup. This is a more modern version of the original transformer archetecture.


## Features

- Utilizing Tri Dao's "flash_attn" library for faster sdpa, also allowing us to sequence pack without slowing down training. So we can train faster on less padding tokens.
- Grouped Query Attention (GQA), to cut down parameters in the multi head attention block while keeping the same reasoning power.
- Rotational Positional Encoding (RoPE), to remove the need for a positional embedding matrix
- Weight Tying, the Token Embedding matrix is also used for the final projection to your logits
- SwiGLU blocks instead of the old ReLU or GeLU feed forward blocks
- Root Mean Squared Norms (RMSNorm)
- Weight initialization based on model size
- Key Value Caching for lightning fast inference
- Inference and training code prints diagnostics to terminal telling you where your time is being spent
- Options to train using fsdp2 or ddp, if you have access to multiple gpus
- Save and load your model checkpoints 
- Utilizes Weights And Biases (WandB) for tracking your runs
- Inference code that can be seeded, and sampling that can plug in a repitition penalty, top-p, top-k and temp


## Data Prep

This training loop is meant to be run on pre tokenized data. Look in the preprocessing folder to find the code I used to tokenize and prep the data for training. You can access a hugging face dataset and then delete it from your hard drive after you save the tokenized data. The tokenized data wont take up as much space on your computer. This will be a lot faster for training then tokenizing it at the same time.

There is a folder for processing pretraining data and one for instruction data. Use the files in order (1 then 2), then use "generate_shuffle.ipynb" to generate a random shuffle for your dataset. Plug in the paths to your data and your shuffle into the training file and you are good to go. Packed sequences will be handled automatically and there will be no cross attention between separate sequences.

I used a custom tokenizer trained using the Hugging Face tokenizers library, the code for that is in the tokenizer folder "hf_tokenizer.ipynb", along with my saved tokenizers which you can use as well if you dont want to train one yourself. 


## Run Training With:

torchrun --nproc_per_node=2 train.py

The "2" can be replaced with how many gpus you have


## Requirements

- cuda-version    13.0   
- flash-attn      2.8.3  
- huggingface-hub 1.16.1 
- numpy           2.2.6  
- python          3.10.20
- tokenizers      0.23.1 
- torch           2.11.0 
- torchao         0.17.0 
- torchaudio      2.11.0 
- torchvision     0.26.0 
- tqdm            4.67.3 
- wandb           0.27.0 
