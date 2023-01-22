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
seed = 990
filename_prefix = 'model15-4-'
batch_size = 64

analyze_corpus = None

csv_filename = 'corpus-snippets.csv'

with torch.inference_mode():
    filename = f'{filename_prefix}{seed}.pickle'
    print(filename)
    with open(filename, 'rb') as f:
        cfg, state = restricted_unpickle(f)
        model = HookedTransformer(cfg)
        model.load_state_dict(state)

    device = cfg.device
    n_ctx = cfg.n_ctx

    
    analyze_corpus = get_corpus(our_seed, n_analyze, train=False, batch_size=batch_size, window_size=n_ctx, tokenizer=model.tokenizer)
    with open(csv_filename, 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for batch in analyze_corpus:
            for sentence in batch:
                tokens = [model.tokenizer.decode(t) for t in sentence]
                writer.writerow(tokens)
