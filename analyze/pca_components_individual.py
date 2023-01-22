import sys
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import math
import csv

component = int(sys.argv[1])   # y

filename = 'data.csv'
prompt_filename = 'corpus-snippets.csv'

n_seeds = 9
n_layers = 1
n_heads = 8
n_ctx = 16
n_analyze = 1024

n_extrema_to_show = 16
n_components = 5
n_analyze_show = 8

with torch.inference_mode():
    n_things = n_seeds * n_layers * n_heads
    n_dimensions = n_analyze * n_ctx * n_ctx

    attn = torch.zeros(n_things, n_dimensions)
    prompts = []

    with open(filename) as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            attn[i] = torch.tensor([float(item) for item in row])

    with open(prompt_filename) as f:
        reader = csv.reader(f)
        for row in reader:
            prompts.append(list(row))

    for seed in range(n_seeds):
        for layer in range(n_layers):
            for head in range(n_heads):
                i = head + n_heads * (layer + n_layers * seed)
                value = seed / n_seeds

    transform = PCA(n_components)
    xy = transform.fit_transform(attn)

    fig, ax = plt.subplots(1, n_analyze_show, squeeze=False)
    for datapoint in range(n_analyze_show):
        grid = transform.components_[component].reshape((n_analyze,n_ctx,n_ctx))[datapoint,:,:]
        ax[0][datapoint].imshow(grid, cmap='coolwarm', vmin=-10/n_analyze, vmax=10/n_analyze)

        print(prompts[datapoint])

        values = [(grid[x,y],x,y) for x in range(n_ctx) for y in range(n_ctx)]
        values.sort(reverse=True)
        for i in list(range(n_extrema_to_show)) + list(range(n_ctx * n_ctx - n_extrema_to_show, n_ctx * n_ctx)):
            v = values[i][0]
            t0 = values[i][2]
            t1 = values[i][1]
            tok0 = prompts[datapoint][t0]
            tok1 = prompts[datapoint][t1]
            print(f'({t0:2},{t1:2}): {v:20} {tok0:10} {tok1:10}')

    plt.show()

