from transformer_lens import HookedTransformer
import sys
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import gc
import math

from corpus import get_corpus
from unp import restricted_unpickle

our_seed = 9992
n_analyze = 1024
seed_min = 990
seed_max = 999
filename_prefix = 'model15-4-'
batch_size = 64

analyze_corpus = None

n_hardcoded = 3
h_colors = [(0,0,0), (0.3,0.3,0.3), (0.75, 0.75, 0.75)]

def make_hardcoded(h, n_ctx):
    global n_analyze
    result = torch.zeros(n_analyze, n_ctx, n_ctx)
    for i in range(n_analyze):
        if h == 0:
            # Everything attends to everything else
            for j in range(n_ctx):
                for k in range(j+1):
                    result[i, j, k] = (1 / (j+1))
        elif h == 1:
            # Everything attends to itself
            for j in range(n_ctx):
                result[i, j, j] = 1
        elif h == 2:
            # Attend to self and prev
            result[i, 0, 0] = 1
            for j in range(1, n_ctx):
                result[i, j, j-1] = 0.5
                result[i, j, j] = 0.5
        else:
            raise Exception(f"No information about hardcoded thing {h}")
    return result.reshape(n_analyze * n_ctx * n_ctx)

with torch.inference_mode():
    for seed in range(seed_min, seed_max):
        filename = f'{filename_prefix}{seed}.pickle'
        print(filename)
        with open(filename, 'rb') as f:
            cfg, state = restricted_unpickle(f)
            model = HookedTransformer(cfg)
            model.load_state_dict(state)

            device = cfg.device
            n_ctx = cfg.n_ctx

            # Stuff we need to delay until we've seen what shape the models are
            if analyze_corpus is None:
                analyze_corpus = get_corpus(our_seed, n_analyze, train=False, batch_size=batch_size, window_size=n_ctx, tokenizer=model.tokenizer)
                attns = torch.zeros((seed_max - seed_min, cfg.n_layers, cfg.n_heads, n_analyze//batch_size, batch_size, n_ctx, n_ctx))
                colors = torch.zeros((seed_max - seed_min, cfg.n_layers, cfg.n_heads, 3))

            for batch_i,sentences in enumerate(analyze_corpus):
                inputs = torch.tensor(sentences).to(device)
                num_things = inputs.shape[0] * n_ctx
                _, cache = model.run_with_cache(inputs)

                for layer in range(cfg.n_layers):
                    for head in range(cfg.n_heads):
                        attn = cache['pattern', layer, 'attn'][:, head, : :]
                        attns[seed - seed_min, layer, head, batch_i, :, :, :] = attn
                        value = (seed - seed_min) / (seed_max - seed_min)
                        colors[seed - seed_min, layer, head, 0] = 0.5 + 0.5 * math.cos(value * 6.283)
                        colors[seed - seed_min, layer, head, 1] = 0.5 + 0.5 * math.cos((value + 0.333) * 6.283)
                        colors[seed - seed_min, layer, head, 2] = 0.5 + 0.5 * math.cos((value + 0.667) * 6.283)

            model = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

    n_things = (seed_max - seed_min) * cfg.n_layers * cfg.n_heads

    attns_plus_h = torch.zeros(n_things + n_hardcoded, n_analyze * n_ctx * n_ctx)
    attns_plus_h[:n_things,:] = attns.reshape(n_things, n_analyze * n_ctx * n_ctx)
    print(attns_plus_h.shape)

    colors_plus_h = torch.zeros((n_things + n_hardcoded, 3))
    colors_plus_h[:n_things,:] = colors.reshape((n_things,3)).to('cpu')
    print(colors_plus_h.shape)

    for h in range(n_hardcoded):
        attns_plus_h[n_things + h, :] = make_hardcoded(h, n_ctx)

        colors_plus_h[n_things + h, 0] = h_colors[h][0]
        colors_plus_h[n_things + h, 1] = h_colors[h][1]
        colors_plus_h[n_things + h, 2] = h_colors[h][2]

    fig, ax = plt.subplots(3,1, squeeze=False)

    transform = umap.UMAP(n_neighbors = 5, min_dist = 0.1, densmap=True)
    xy = transform.fit_transform(attns_plus_h)
    ax[0][0].scatter(xy[:,0], xy[:,1], c=colors_plus_h)

    transform = TSNE(2, perplexity = 5)
    xy = transform.fit_transform(attns_plus_h)
    ax[1][0].scatter(xy[:,0], xy[:,1], c=colors_plus_h)

    transform = PCA(2)
    xy = transform.fit_transform(attns_plus_h)
    ax[2][0].scatter(xy[:,0], xy[:,1], c=colors_plus_h)
    plt.show()

