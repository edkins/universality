import sys
import torch
from matplotlib import pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import math
import csv
import re

filename = 'data.csv'
prompt_filename = 'corpus-snippets.csv'

datapoint = int(sys.argv[1])

n_seeds = 9
n_layers = 1
n_heads = 8
n_ctx = 16
n_analyze = 1024

with torch.inference_mode():
    n_things = n_seeds * n_layers * n_heads
    n_dimensions = n_analyze * n_ctx * n_ctx

    attn = torch.zeros(n_things, n_dimensions)
    prompts = []

    with open(filename) as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            attn[i] = torch.tensor([float(item) for item in row])

    attn = attn.reshape(n_seeds, n_layers * n_heads, n_analyze, n_ctx, n_ctx)

    with open(prompt_filename) as f:
        reader = csv.reader(f)
        for row in reader:
            prompts.append(list(row))

    print(prompts[datapoint])

    fig, ax = plt.subplots(n_layers * n_heads, n_seeds, squeeze=False)
    for seed in range(n_seeds):
        for head in range(n_layers * n_heads):
            ax[head][seed].imshow(attn[seed,head,datapoint,:,:])

    plt.show()

