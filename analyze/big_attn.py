import sys
import torch
from matplotlib import pyplot as plt
import csv

filename = 'data.csv'

n_seeds = 9
n_layers = 1
n_heads = 8
n_ctx = 16
n_analyze = 1024

n_things_show = 16
n_analyze_show = 16

with torch.inference_mode():
    n_things = n_seeds * n_layers * n_heads
    n_dimensions = n_analyze * n_ctx * n_ctx
    grid = torch.zeros((n_things, n_things))

    attn = torch.zeros(n_things, n_dimensions)

    with open(filename) as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            print(i)
            attn[i] = torch.tensor([float(item) for item in row])

    attn = attn.reshape((n_things, n_analyze, n_ctx, n_ctx))

    fig, ax = plt.subplots(n_things_show, n_analyze_show, squeeze=False)
    for thing in range(n_things_show):
        for datapoint in range(n_analyze_show):
            print(thing, datapoint)
            ax[thing][datapoint].imshow(attn[thing, datapoint, :, :])

    plt.show()

