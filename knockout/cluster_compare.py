from transformer_lens import HookedTransformer
import sys
import torch
import random
import csv
import matplotlib.pyplot as plt
import gc

from unp import restricted_unpickle

min_seed = 990

cluster_indices = [2,6,1,3,4,2,5,1,1,6,5,1,7,3,4,2,4,7,2,3,5,6,1,7,3,5,2,2,1,7,6,4,2,5,4,1,3,5,7,7,7,3,4,1,4,6,2,3,2,1,6,3,4,1,6,6,4,2,1,7,6,3,5,1,5,4,5,3,1,2,1,1]

n_heads_per_model = 8
n_rows = 3
n_cols = 4

with torch.inference_mode():
    cluster_index = int(sys.argv[1])
    prompt = sys.argv[2]
        
    fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)

    for row in range(n_rows):
        for col in range(n_cols):
            ax[row][col].set_axis_off()

    row = 0
    col = 0
    for modelhead,index in enumerate(cluster_indices):
        if index != cluster_index:
            continue

        seed = min_seed + (modelhead // n_heads_per_model)
        knockout_head = modelhead % n_heads_per_model

        print(f"Model {seed}, head {knockout_head}")

        with open(f'model15-4-{seed}.pickle', 'rb') as f:
            cfg, state = restricted_unpickle(f)
            model = HookedTransformer(cfg)
            model.load_state_dict(state)

        def my_hook_attn_z(activation, hook):
            a, b, _, c = activation.shape
            activation[:, :, knockout_head, :] = torch.zeros((a,b,c))
            return activation

        tokens = model.tokenizer.encode(prompt)
        print(tokens)
        last = len(tokens) - 1
        ttokens = torch.tensor([tokens])
        logits = model(ttokens).cpu()
        logitsk = model.run_with_hooks(ttokens, fwd_hooks=[('blocks.0.attn.hook_z', my_hook_attn_z)]).cpu()

        values = [(v.item(),j) for j,v in enumerate(logits[0,last,:])]
        values.sort(reverse=True)

        xs = []
        ys = []
        for _,j in values[:30]:
            x = logits[0,last,j].item()
            y = logitsk[0,last,j].item()
            ax[row][col].text(x, y, model.tokenizer.decode(j), fontsize='xx-small')
            xs.append(x)
            ys.append(y)

        ax[row][col].scatter(xs, ys, s=1)
        ax[row][col].set_title(f"Model {seed - min_seed + 1}, head {knockout_head + 1}")

        col += 1
        if col >= n_cols:
            col = 0
            row += 1
            if row >= n_rows:
                break

        model = None
        gc.collect()

    fig.suptitle(f'[cluster {cluster_index}] {prompt}')
    plt.show()
