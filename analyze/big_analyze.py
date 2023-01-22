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

with torch.inference_mode():
    n_things = n_seeds * n_layers * n_heads
    n_dimensions = n_analyze * n_ctx * n_ctx
    grid = torch.zeros((n_things, n_things))

    attn = torch.zeros(n_things, n_dimensions)

    with open(filename) as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            attn[i] = torch.tensor([float(item) for item in row])

    for x in range(n_things):
        for y in range(n_things):
            grid[x,y] = (attn[x] - attn[y]).norm()

    plt.imshow(grid.cpu())
    plt.show()

