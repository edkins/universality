from transformer_lens import HookedTransformer
import sys
import torch
from matplotlib import pyplot as plt

from unp import restricted_unpickle


filename = sys.argv[1]
prompt = sys.argv[2]

with open(filename, 'rb') as f:
    cfg, state = restricted_unpickle(f)
model = HookedTransformer(cfg)
model.load_state_dict(state)

with torch.inference_mode():
    tokens = model.tokenizer.encode(prompt)[:cfg.n_ctx]
    print([model.tokenizer.decode(t) for t in tokens])
    inputs = torch.tensor([tokens])
    _, cache = model.run_with_cache(inputs)

    fix, ax = plt.subplots(cfg.n_layers, cfg.n_heads)
    for layer in range(cfg.n_layers):
        for head in range(cfg.n_heads):
            attn = cache['pattern', layer, 'attn'][0, head, : :]
            ax[layer][head].imshow(attn.cpu())
    plt.show()
