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

n_components = 5
n_analyze_show = 8

with torch.inference_mode():
    n_things = n_seeds * n_layers * n_heads
    n_dimensions = n_analyze * n_ctx * n_ctx

    attn = torch.zeros(n_things, n_dimensions)

    with open(filename) as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            attn[i] = torch.tensor([float(item) for item in row])

    for seed in range(n_seeds):
        for layer in range(n_layers):
            for head in range(n_heads):
                i = head + n_heads * (layer + n_layers * seed)
                value = seed / n_seeds

    transform = PCA(n_components)
    xy = transform.fit_transform(attn)
    #ax[0][0].scatter(xy[:,0], xy[:,1])

    fig, ax = plt.subplots(n_components, n_analyze_show, squeeze=False)
    for component in range(n_components):
        for datapoint in range(n_analyze_show):
            grid = transform.components_[component].reshape((n_analyze,n_ctx,n_ctx))[datapoint,:,:]
            ax[component][datapoint].imshow(grid, cmap='coolwarm', vmin=-10/n_analyze, vmax=10/n_analyze)

    plt.show()

