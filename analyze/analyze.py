from transformer_lens import HookedTransformer
import sys
import torch
from matplotlib import pyplot as plt

from corpus import get_corpus
from unp import restricted_unpickle

seed = 9992
n_analyze = 10000

filename0 = sys.argv[1]
filename1 = sys.argv[2]
with open(filename0, 'rb') as f:
    cfg0, state0 = restricted_unpickle(f)
with open(filename1, 'rb') as f:
    cfg1, state1 = restricted_unpickle(f)

model0 = HookedTransformer(cfg0)
model0.load_state_dict(state0)

model1 = HookedTransformer(cfg1)
model1.load_state_dict(state1)

device = cfg0.device
n_ctx = cfg0.n_ctx
analyze_corpus = get_corpus(seed, n_analyze, train=False, batch_size=64, window_size=n_ctx, tokenizer=model0.tokenizer)

distances = torch.zeros((cfg0.n_layers * cfg0.n_heads, cfg1.n_layers * cfg1.n_heads)).to(device)
with torch.inference_mode():
    for sentences in analyze_corpus:
        inputs = torch.tensor(sentences).to(device)
        num_things = inputs.shape[0] * n_ctx
        _, cache0 = model0.run_with_cache(inputs)
        _, cache1 = model1.run_with_cache(inputs)

        for layer0 in range(cfg0.n_layers):
            for layer1 in range(cfg0.n_layers):
                for head0 in range(cfg1.n_heads):
                    for head1 in range(cfg1.n_heads):
                        attn0 = cache0['pattern', layer0, 'attn'][:, head0, : :]
                        attn1 = cache1['pattern', layer1, 'attn'][:, head1, :,:]
                        diff = attn1 - attn0
                        distance_sq = (diff * diff).sum()
                        x = layer0 * cfg0.n_heads + head0
                        y = layer1 * cfg1.n_heads + head1
                        distances[x,y] += distance_sq

plt.imshow(torch.sqrt(distances).cpu())
plt.show()

