from transformer_lens import HookedTransformer, HookedTransformerConfig

from corpus import get_corpus

n_layers = 3
n_heads = 4
n_ctx = 16  # The maximum sequence length
seed = 999

#https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#autotokenizer
tokenizer_name = 'gpt2'

cfg = HookedTransformerConfig(
    n_layers = n_layers,
    n_heads = n_heads,
    d_model = 128,
    d_head = 32,
    d_mlp = 512,
    act_fn = "relu",
    normalization_type=None,
    tokenizer_name=tokenizer_name,
    n_ctx = n_ctx,
    init_weights=True,
    device="cuda",
    seed = seed,
)

model = HookedTransformer(cfg)
for s in get_corpus(seed, 100):
    print(s)

