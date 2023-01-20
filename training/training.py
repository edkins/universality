from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm
import torch

from corpus import get_corpus

n_layers = 3
n_heads = 4
n_ctx = 16  # The maximum sequence length
seed = 999

#https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#autotokenizer
tokenizer_name = 'gpt2'

num_epochs = 20
n_corpus = 1000

device = 'cuda'

######

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
    device = device,
    seed = seed,
)

model = HookedTransformer(cfg)
d_vocab = model.cfg.d_vocab

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in tqdm.tqdm(range(num_epochs)):
    loss_sum = torch.zeros(()).to(device)
    for sentences in get_corpus(seed, n_corpus, train=True, batch_size=64, window_size=n_ctx+1, tokenizer=model.tokenizer):
        stensor = torch.tensor(sentences).to(device)
        num_things = stensor.shape[0] * n_ctx
        nexts = stensor[:,1:].reshape(num_things)
        inputs = stensor[:,:-1]
        
        logits = model(inputs).reshape(num_things, d_vocab)
        loss = loss_fn(logits, nexts)
        loss.backward()
        loss_sum += loss.detach()
        
        optimizer.step()
        optimizer.zero_grad()
    print(f'Epoch {epoch}: train loss = {loss_sum.item()}')

