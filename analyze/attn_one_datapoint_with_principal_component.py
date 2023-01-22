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
component = int(sys.argv[2])

n_seeds = 9
n_layers = 1
n_heads = 8
n_ctx = 16
n_analyze = 1024

n_components = 5

with torch.inference_mode():
    n_things = n_seeds * n_layers * n_heads
    n_dimensions = n_analyze * n_ctx * n_ctx

    attn = torch.zeros(n_things, n_dimensions)
    prompts = []

    with open(filename) as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            attn[i] = torch.tensor([float(item) for item in row])

    transform = PCA(n_components)
    pca = transform.fit_transform(attn)[:, component-1].reshape(n_seeds, n_layers * n_heads)

    attn = attn.reshape(n_seeds, n_layers * n_heads, n_analyze, n_ctx, n_ctx)

    with open(prompt_filename) as f:
        reader = csv.reader(f)
        for row in reader:
            prompts.append(list(row))

    print(prompts[datapoint])

    fig, ax = plt.subplots(n_layers * n_heads, n_seeds, squeeze=False)
    for seed in range(n_seeds):
        for head in range(n_layers * n_heads):
            colors = torch.zeros((n_ctx, n_ctx, 3))
            colors[:,:,1] = attn[seed,head,datapoint,:,:]
            colors[:,:,0] = 0.5 + 0.5 * math.tanh(pca[seed,head] / 40)
            ax[head][seed].imshow(colors)

    plt.show()

