import sys
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import math
import csv

filename = 'data.csv'

n_seeds = 9
n_layers = 1
n_heads = 8
n_ctx = 16
n_analyze = 1024

with torch.inference_mode():
    n_things = n_seeds * n_layers * n_heads
    n_dimensions = n_analyze * n_ctx * n_ctx

    attn = torch.zeros(n_things, n_dimensions)

    with open(filename) as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            attn[i] = torch.tensor([float(item) for item in row])

    colors = torch.zeros((n_things, 3))
    for seed in range(n_seeds):
        for layer in range(n_layers):
            for head in range(n_heads):
                i = head + n_heads * (layer + n_layers * seed)
                value = seed / n_seeds
                colors[i,0] = 0.5 + 0.5 * math.cos(value * 6.283)
                colors[i,1] = 0.5 + 0.5 * math.cos((value + 0.333) * 6.283)
                colors[i,2] = 0.5 + 0.5 * math.cos((value + 0.667) * 6.283)

    fig, ax = plt.subplots(3,1, squeeze=False)

    transform = umap.UMAP(n_neighbors = 5, min_dist = 0.1, densmap=True)
    xy = transform.fit_transform(attn)
    ax[0][0].scatter(xy[:,0], xy[:,1], c=colors)

    transform = TSNE(2, perplexity = 5)
    xy = transform.fit_transform(attn)
    ax[1][0].scatter(xy[:,0], xy[:,1], c=colors)

    transform = PCA(2)
    xy = transform.fit_transform(attn)
    ax[2][0].scatter(xy[:,0], xy[:,1], c=colors)
    plt.show()

