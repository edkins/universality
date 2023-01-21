from transformer_lens import HookedTransformer
import pickle
import sys
import readline
import torch
import random

temperature = 0.5

def complete(model, prompt, temperature):
    tokens = model.tokenizer.encode(prompt)
    
    for _ in range(10):
        logits = model(torch.tensor([tokens[-n_ctx:]]))[0,-1,:]
        probs = torch.nn.functional.softmax(logits / temperature, dim=0)
        d_vocab = probs.shape[0]

        next_token = random.choices(range(d_vocab), weights=probs)[0]
        tokens.append(next_token)
    return model.tokenizer.decode(tokens)

filename = sys.argv[1]
with open(filename, 'rb') as f:
    cfg, state = pickle.load(f)

model = HookedTransformer(cfg)
model.load_state_dict(state)

n_ctx = model.cfg.n_ctx

prompt = input('Prompt> ')
for i in range(10):
    print(complete(model, prompt, temperature))

