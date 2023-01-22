import sys
import torch
from matplotlib import pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import umap
import math
import csv
import re

filename = 'scores.csv'
prompt_filename = 'corpus-snippets.csv'

n_seeds = 9
n_layers = 1
n_heads = 8
n_ctx = 16
n_analyze = 1024

n_analyze_show = 32

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

    with open(prompt_filename) as f:
        reader = csv.reader(f)
        for row in reader:
            prompts.append(list(row))

    X = torch.zeros((n_analyze_show, n_ctx, n_things))
    for i in range(n_analyze_show):
        for j in range(n_ctx):
            X[i, j, :] = attn[:, i, j, j]

    X = X.reshape((n_analyze_show * n_ctx, n_things))

    pca = PCA(2)
    xy = pca.fit_transform(X)
    plt.scatter(xy[:,0], xy[:,1], s=1)
    for i in range(n_analyze_show * n_ctx):
        plt.text(xy[i,0], xy[i,1], prompts[i // n_ctx][i % n_ctx], fontsize='xx-small')
    plt.show()


