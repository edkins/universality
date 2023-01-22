from transformer_lens import HookedTransformer
import sys
import torch
import gc
import math
import csv

from corpus import get_corpus
from unp import restricted_unpickle

our_seed = 9992
n_analyze = 1024
seed_min = 990
seed_max = 999
filename_prefix = 'model15-4-'
batch_size = 64

analyze_corpus = None

csv_filename = 'scores.csv'

with torch.inference_mode():
    for seed in range(seed_min, seed_max):
        filename = f'{filename_prefix}{seed}.pickle'
        print(filename)
        with open(filename, 'rb') as f:
            cfg, state = restricted_unpickle(f)
            model = HookedTransformer(cfg)
            model.load_state_dict(state)

            device = cfg.device
            n_ctx = cfg.n_ctx

            # Stuff we need to delay until we've seen what shape the models are
            if analyze_corpus is None:
                analyze_corpus = get_corpus(our_seed, n_analyze, train=False, batch_size=batch_size, window_size=n_ctx, tokenizer=model.tokenizer)
                attns = torch.zeros((seed_max - seed_min, cfg.n_layers, cfg.n_heads, n_analyze//batch_size, batch_size, n_ctx, n_ctx))

            for batch_i,sentences in enumerate(analyze_corpus):
                inputs = torch.tensor(sentences).to(device)
                num_things = inputs.shape[0] * n_ctx
                _, cache = model.run_with_cache(inputs)

                for layer in range(cfg.n_layers):
                    for head in range(cfg.n_heads):
                        attn = cache['attn_scores', layer, 'attn'][:, head, : :]
                        attns[seed - seed_min, layer, head, batch_i, :, :, :] = attn
                        value = (seed - seed_min) / (seed_max - seed_min)

            model = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

with open(csv_filename, 'w') as f:
    writer = csv.writer(f)
    data = attns.reshape((seed_max - seed_min) * cfg.n_layers * cfg.n_heads, n_analyze * n_ctx * n_ctx).to('cpu')
    for i,row in enumerate(data):
        print(i, len(data))
        writer.writerow([x.item() for x in row])

