from transformer_lens import HookedTransformer
import sys
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import umap
import gc

from corpus import get_corpus
from unp import restricted_unpickle

our_seed = 9992
n_analyze = 1024
seed_min = 990
seed_max = 1000
filename_prefix = 'model15-4-'
batch_size = 64

analyze_corpus = None

with torch.inference_mode():
    for seed in range(seed_min, seed_max):
        filename = f'{filename_prefix}{seed}.pickle'
        print(filename)
        with open(filename, 'rb') as f:
            cfg, state = restricted_unpickle(f)
            model = HookedTransformer(cfg)

            device = cfg.device
            n_ctx = cfg.n_ctx

            # Stuff we need to delay until we've seen what shape the models are
            if analyze_corpus is None:
                analyze_corpus = get_corpus(our_seed, n_analyze, train=False, batch_size=batch_size, window_size=n_ctx, tokenizer=model.tokenizer)
                attns = torch.zeros((seed_max - seed_min, cfg.n_layers, cfg.n_heads, n_analyze//batch_size, batch_size, n_ctx, n_ctx))
                colors = torch.zeros((seed_max - seed_min, cfg.n_layers, cfg.n_heads))

            for batch_i,sentences in enumerate(analyze_corpus):
                inputs = torch.tensor(sentences).to(device)
                num_things = inputs.shape[0] * n_ctx
                _, cache = model.run_with_cache(inputs)

                for layer in range(cfg.n_layers):
                    for head in range(cfg.n_heads):
                        attn = cache['pattern', layer, 'attn'][:, head, : :]
                        attns[seed - seed_min, layer, head, batch_i, :, :, :] = attn
                        colors[seed - seed_min, layer, head] = seed - seed_min

            model = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

    n_things = (seed_max - seed_min) * cfg.n_layers * cfg.n_heads
    attns = attns.reshape(n_things, n_analyze * n_ctx * n_ctx)
    print(attns.shape)

    colors = colors.reshape((n_things,)).to('cpu')
    print(colors.shape)

    trans = umap.UMAP(n_neighbors = 2, min_dist = 0.1)
    #trans = TSNE(2, perplexity = 5)
    xy = trans.fit_transform(attns)

    plt.scatter(xy[:,0], xy[:,1], c=colors)
    plt.show()

