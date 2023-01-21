from transformer_lens import HookedTransformer, HookedTransformerConfig
import tqdm
import torch
import pickle
import random

from corpus import get_corpus

from optparse import OptionParser


parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",default='model',
                  help="write model to FILE", metavar="FILE")
parser.add_option("-e", "--epochs",
                  action="store", type='int',dest="epochs", default=50,
                  help="number of epochs")

(options, args) = parser.parse_args()

n_layers = 2
n_heads = 2
n_ctx = 16 # The maximum sequence length
seed = 999
filename = options.filename + str(options.epochs) + '.pickle'

#https://huggingface.co/transformers/v3.0.2/model_doc/auto.html#autotokenizer
tokenizer_name = 'gpt2'

num_epochs = options.epochs
n_train = 30000
n_test = 1000

device = 'cuda'

######

cfg = HookedTransformerConfig(
    n_layers = n_layers,
    n_heads = n_heads,
    d_model = 64,
    d_head = 32,
    d_mlp = 128,
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

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1, betas=(0.9, 0.98))
loss_fn = torch.nn.CrossEntropyLoss()

train_corpus = get_corpus(seed, n_train, train=True, batch_size=64, window_size=n_ctx+1, tokenizer=model.tokenizer)
test_corpus = get_corpus(seed, n_test, train=False, batch_size=64, window_size=n_ctx+1, tokenizer=model.tokenizer)

random.seed(seed)
for epoch in tqdm.tqdm(range(num_epochs)):
    try:
        loss_sum = torch.zeros(()).to(device)
        n_batches = 0
        random.shuffle(train_corpus)
        for sentences in train_corpus:
            stensor = torch.tensor(sentences).to(device)
            num_things = stensor.shape[0] * n_ctx
            nexts = stensor[:,1:].reshape(num_things)
            inputs = stensor[:,:-1]
            
            logits = model(inputs).reshape(num_things, d_vocab)
            loss = loss_fn(logits, nexts)
            loss.backward()
            loss_sum += loss.detach()
            n_batches += 1
            
            optimizer.step()
            optimizer.zero_grad()
        print(f'Epoch {epoch}: train loss = {loss_sum.item() / n_batches}')

        loss_sum = torch.zeros(()).to(device)
        n_batches = 0
        with torch.inference_mode():
            for sentences in test_corpus:
                stensor = torch.tensor(sentences).to(device)
                num_things = stensor.shape[0] * n_ctx
                nexts = stensor[:,1:].reshape(num_things)
                inputs = stensor[:,:-1]
                
                logits = model(inputs).reshape(num_things, d_vocab)
                loss_sum += loss_fn(logits, nexts)
                n_batches += 1
        print(f'         test loss = {loss_sum.item() / n_batches}')
    finally:
        with open(filename, 'wb') as f:
            pickle.dump((cfg, model.state_dict()), f)


