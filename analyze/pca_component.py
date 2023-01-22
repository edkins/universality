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

component = int(sys.argv[1])

filename = 'data.csv'
prompt_filename = 'corpus-snippets.csv'

n_seeds = 9
n_layers = 1
n_heads = 8
n_ctx = 16
n_analyze = 1024

n_components = 5
n_analyze_show = 8
n_columns = 2

show_punc = False

re_partial = re.compile(r'^[a-z].*$')
def is_partial(token):
    return re_partial.match(token) is not None

re_punc = re.compile(r'^[^a-zA-Z ].*$')
def is_punc(token):
    return re_punc.match(token) is not None

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

    for datapoint in range(n_analyze_show):
        print(datapoint, prompts[datapoint])

    fig, ax = plt.subplots(n_analyze_show // n_columns, n_columns, squeeze=False)
    cmap = matplotlib.colormaps['coolwarm']
    for datapoint in range(n_analyze_show):
        grid = transform.components_[component].reshape((n_analyze,n_ctx,n_ctx))[datapoint,:,:]
        colors = torch.zeros((n_ctx, n_ctx, 3))
        for x in range(n_ctx):
            for y in range(n_ctx):   # oops: x and y are the wrong way around
                if x >= y:
                    colors[x,y,:] = torch.tensor(cmap(0.5 + grid[x,y] * n_analyze / 20)[:3])
                elif show_punc and is_partial(prompts[datapoint][y]):
                    colors[x,y,0] = 0.8764
                    colors[x,y,1] = 1
                    colors[x,y,2] = 0.8764
                elif show_punc and is_punc(prompts[datapoint][y]):
                    colors[x,y,0] = 0.8764
                    colors[x,y,1] = 1
                    colors[x,y,2] = 1
                else:
                    colors[x,y,0] = 0.8764
                    colors[x,y,1] = 0.8764
                    colors[x,y,2] = 0.8764
        ax[datapoint // n_columns][datapoint % n_columns].imshow(colors)
        for word_index in range(n_ctx):
            ax[datapoint // n_columns][datapoint % n_columns].text(n_ctx, word_index, prompts[datapoint][word_index], fontsize='xx-small')

    plt.show()

