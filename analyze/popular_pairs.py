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

n_seeds = 9
n_layers = 1
n_heads = 8
n_ctx = 16
n_analyze = 1024

n_analyze_show = 8
threshold = 0.2
n_columns = 2

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

    for seed in range(n_seeds):
        for layer in range(n_layers):
            for head in range(n_heads):
                i = head + n_heads * (layer + n_layers * seed)
                value = seed / n_seeds

    for datapoint in range(n_analyze_show):
        print(datapoint, prompts[datapoint])

    fig, ax = plt.subplots(n_analyze_show // n_columns, n_columns, squeeze=False)
    cmap = matplotlib.colormaps['coolwarm']
    for datapoint in range(n_analyze_show):
        grid = (attn[:, datapoint, :, :] > threshold).sum(dim=0)
        ax[datapoint // n_columns][datapoint % n_columns].imshow(grid)
        for word_index in range(n_ctx):
            ax[datapoint // n_columns][datapoint % n_columns].text(n_ctx, word_index, prompts[datapoint][word_index], fontsize='xx-small')

    plt.show()

