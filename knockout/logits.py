from transformer_lens import HookedTransformer
import sys
import torch
import random
import csv
import matplotlib.pyplot as plt

from unp import restricted_unpickle

prompt_filename = 'corpus-snippets.csv'
num_prompts_to_use = 4

with torch.inference_mode():
    filename = sys.argv[1]
    knockout_head = int(sys.argv[2])

    with open(filename, 'rb') as f:
        cfg, state = restricted_unpickle(f)

    model = HookedTransformer(cfg)
    model.load_state_dict(state)

    print(model.hook_dict.keys())

    def my_hook_attn_scores(activation, hook):
        a, _, b, c = activation.shape
        for i in range(c):
            activation[:, knockout_head, i, :i+1] = torch.zeros((a, i+1))
        return activation

    def my_hook_attn_z(activation, hook):
        a, b, _, c = activation.shape
        activation[:, :, knockout_head, :] = torch.zeros((a,b,c))
        return activation

    prompts = []
    with open(prompt_filename) as f:
        reader = csv.reader(f)
        for row in reader:
            prompts.append(list(row))

    for prompt in prompts[:num_prompts_to_use]:
        print(prompt)
        tokens = [model.tokenizer.encode(t)[0] for t in prompt]
        print(tokens)
        ttokens = torch.tensor([tokens])
        logits = model(ttokens).cpu()
        #logits2 = model.run_with_hooks(ttokens, fwd_hooks=[('blocks.0.attn.hook_attn_scores', my_hook)]).cpu()
        logits2 = model.run_with_hooks(ttokens, fwd_hooks=[('blocks.0.attn.hook_z', my_hook_attn_z)]).cpu()
        
        for i in range(len(prompt)):
            print(prompt[i])
            values = [(v.item(),model.tokenizer.decode(j)) for j,v in enumerate(logits[0,i,:])]
            values.sort(reverse=True)
            print(values[:5])
            values = [(v.item(),model.tokenizer.decode(j)) for j,v in enumerate(logits2[0,i,:])]
            values.sort(reverse=True)
            print(values[:5])

            plt.title(''.join(prompt[:i+1]))
            plt.scatter(logits[0,i,:], logits2[0,i,:], s=1)

            values = [(v.item(),j) for j,v in enumerate(logits[0,i,:])]
            values.sort(reverse=True)

            for _,j in values[:30]:
                plt.text(logits[0,i,j].item(), logits2[0,i,j].item(), model.tokenizer.decode(j), fontsize='xx-small')

            plt.show()
