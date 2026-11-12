import torch
from tokenizers import Tokenizer
import gc
from model import GPT, GPTConfig

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running inference on: {device}")

tokenizer_file = "tokenizer/tokenizer_02.json"
model_id = "02"
model_step = "42300"
data_abreviation = "pre"
load_path = f'checkpoints/m{model_id}/model{model_id}_{data_abreviation}{model_step}.pt'

dtype = torch.bfloat16
max_new_tokens = 512
rep_pen = 1.1
temp = 0.8
topk = 50
topp = 0.9
seed = 2
context_len = 8192

tokenizer = Tokenizer.from_file(tokenizer_file)

# LOAD a model from checkpoint
m = GPT(GPTConfig()).to(device).to(dtype)
checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
m.load_state_dict(checkpoint['model'])

print(f"\nModel loaded successfuly from {load_path}")
print(f"Total parameters: {sum(p.numel() for p in m.parameters()):,}")
print(f"Model Vocab Size: {m.config.vocab_size}")
print(f"Tokenizer Vocab Size: {tokenizer.get_vocab_size()}\n")

del checkpoint
gc.collect()
torch.cuda.empty_cache()

text_complete = 1
if text_complete:
    # Text completion
    prompts = [
        "The heavy, iron door groaned as it swung open, revealing a room that hadn't seen light in centuries. The very first thing I noticed was",
        "The Apollo 11 mission was a historic milestone in human spaceflight. On July 20, 1969, astronauts Neil Armstrong and Buzz Aldrin became the first two humans to",
        #"There are three primary reasons why electric vehicles are gaining market share. First, they are much better for the environment. Second,",
        #"\"I cannot believe you actually managed to fix the engine,\" Sarah said, wiping the grease from her forehead. John smirked, tossed his wrench onto the workbench, and replied,",
        #"While many economists argue that artificial intelligence will completely replace human workers, a growing number of researchers believe the exact opposite; they argue that AI will actually"
        ]
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        
        ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        context = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.amp.autocast('cuda', dtype=dtype):
            out = m.generate(context, max_new_tokens=max_new_tokens, rep_pen=rep_pen, temp=temp, top_k=topk, top_p=topp, seed=seed)
    
        print(f"\nLLouis 1B:")
        print(tokenizer.decode(out, skip_special_tokens=True))
        print()
        print("-" * 50)
else:
    # CHATBOT
    context = torch.empty((1, 0), dtype=torch.long, device=device)
    print("LLouis 1B is ready to chat! (Type 'exit' to quit)\n" + "-" * 50)
    while True:
        user = input("User: ")
        if user.lower() in ["exit", "quit", "q"]:
            print("\nLLouis 1B: Thanks for chatting with me today! Goodbye!")
            print("-" * 50)
            print("Exiting chatbot...")
            break

        ids = torch.tensor([tokenizer.encode(user, add_special_tokens=True).ids], dtype=torch.long, device=device)
        context = torch.cat((context, ids), dim=1)
        print(f"\nContext: {tokenizer.decode(context[0].tolist(), skip_special_tokens=False)}")
        with torch.amp.autocast('cuda', dtype=dtype):
            out = m.generate(context, max_new_tokens=max_new_tokens, rep_pen=rep_pen, temp=temp, top_k=topk, top_p=topp, seed=seed)

        print(f"\nLLouis 1B: {tokenizer.decode(out, skip_special_tokens=True)}")
        print("-" * 50)

        context = torch.empty((1, 0), dtype=torch.long, device=device) #torch.cat((context, torch.tensor([out], dtype=torch.long, device=device), dim=1)[:, -(context_len-max_new_tokens):]