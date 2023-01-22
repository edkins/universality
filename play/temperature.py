from transformer_lens import HookedTransformer
import sys
import readline
import torch
import random

from unp import restricted_unpickle

def complete(model, prompt, temperature):
    tokens = model.tokenizer.encode(prompt)
    
    for _ in range(10):
        logits = model(torch.tensor([tokens[-n_ctx:]]))[0,-1,:]
        probs = torch.nn.functional.softmax(logits / temperature, dim=0)
        d_vocab = probs.shape[0]

        next_token = list(torch.utils.data.WeightedRandomSampler(probs, 1, replacement=True))[0]
        tokens.append(next_token)
    return model.tokenizer.decode(tokens)

filename = sys.argv[1]
with open(filename, 'rb') as f:
    cfg, state = restricted_unpickle(f)

if len(sys.argv) >= 3:
    temperature = float(sys.argv[2])
else:
    temperature = 0.5

model = HookedTransformer(cfg)
model.load_state_dict(state)

n_ctx = model.cfg.n_ctx

prompt = input('Prompt> ')
for i in range(10):
    print(complete(model, prompt, temperature))

