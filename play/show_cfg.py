from transformer_lens import HookedTransformer
import sys
import readline
import torch
import random

from unp import restricted_unpickle

filename = sys.argv[1]
with open(filename, 'rb') as f:
    cfg, state = restricted_unpickle(f)

print(cfg)
