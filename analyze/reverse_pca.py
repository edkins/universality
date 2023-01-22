import sys
import torch
from matplotlib import pyplot as plt
import matplotlib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import math
import csv
import re

filename = 'data.csv'
prompt_filename = 'corpus-snippets.csv'

n_seeds = 9
n_layers = 1
n_heads = 8
n_ctx = 16
n_analyze = 1024

n_analyze_show = 64

with torch.inference_mode():
    n_things = n_seeds * n_layers * n_heads
    n_dimensions = n_analyze * n_ctx * n_ctx

    attn = torch.zeros(n_things, n_dimensions)
    prompts = []

    with open(filename) as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            attn[i] = torch.tensor([float(item) for item in row])

    attn = attn.reshape((n_things, n_analyze, n_ctx, n_ctx))
    X = torch.zeros((n_analyze_show * (n_ctx * (n_ctx + 1) // 2 - 1), n_things))
    index = 0
    for i in range(n_analyze_show):
        for j in range(1, n_ctx):
            for k in range(j+1):
                X[index,:] = attn[:, i, j, k]
                index += 1

    with open(prompt_filename) as f:
        reader = csv.reader(f)
        for row in reader:
            prompts.append(list(row))

    #transform = UMAP(n_neighbors=15, min_dist = 0.1, densmap=True)
    transform = PCA(2)
    xy = transform.fit_transform(X)

    fig, ax = plt.subplots(1, 1, squeeze=False)
    ax[0][0].scatter(xy[:,0], xy[:,1], s=1)
    plt.show()

