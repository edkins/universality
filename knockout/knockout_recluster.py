from transformer_lens import HookedTransformer
import sys
import torch
import random
import math
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
import gc

from unp import restricted_unpickle

min_seed = 990

original_cluster_indices = [2,6,1,3,4,2,5,1,1,6,5,1,7,3,4,2,4,7,2,3,5,6,1,7,3,5,2,2,1,7,6,4,2,5,4,1,3,5,7,7,7,3,4,1,4,6,2,3,2,1,6,3,4,1,6,6,4,2,1,7,6,3,5,1,5,4,5,3,1,2,1,1]
num_clusters = max(original_cluster_indices)

n_models = 9
n_heads_per_model = 8

with torch.inference_mode():
    prompt = sys.argv[1]
        
    fig, ax = plt.subplots(1, 1, squeeze=False)

    ax[0][0].set_axis_off()

    X = None

    colormap = torch.tensor([
        [0,0,0],
        [97,46,42],
        [77,60,0],
        [33,71,0],
        [0,75,58],
        [0,71,92],
        [65,54,100],
        [98,38,84],
        ]) / 100
    colors = torch.zeros((n_models * n_heads_per_model,3))
    for modelid in range(n_models):
        seed = min_seed + modelid
        with open(f'model15-4-{seed}.pickle', 'rb') as f:
            cfg, state = restricted_unpickle(f)
            model = HookedTransformer(cfg)
            model.load_state_dict(state)

        if X is None:
            X = torch.zeros((n_models * n_heads_per_model, model.cfg.d_vocab))

        tokens = model.tokenizer.encode(prompt)
        last = len(tokens) - 1
        ttokens = torch.tensor([tokens])
        probs = torch.nn.functional.softmax(model(ttokens).cpu(), dim=1)

        for knockout_head in range(n_heads_per_model):
            modelhead = knockout_head + n_heads_per_model * modelid

            print(f"Model {seed}, head {knockout_head}")

            def my_hook_attn_z(activation, hook):
                a, b, _, c = activation.shape
                activation[:, :, knockout_head, :] = torch.zeros((a,b,c))
                return activation

            probsk = torch.nn.functional.softmax(
                    model.run_with_hooks(ttokens, fwd_hooks=[('blocks.0.attn.hook_z', my_hook_attn_z)]).cpu(),
                    dim=1
            )
            diffs = probsk - probs
            X[modelhead, :] = diffs[0, last, :]

            value = original_cluster_indices[modelhead] / num_clusters
            colors[modelhead,:] = colormap[original_cluster_indices[modelhead]]

    xy = UMAP(n_neighbors=5, min_dist=0.1, densmap=True).fit_transform(X)

    ax[0][0].scatter(xy[:,0], xy[:,1], c=colors)
    ax[0][0].legend()
    fig.suptitle(prompt)
    plt.show()
