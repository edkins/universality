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
    filename0 = sys.argv[1]
    knockout_head0 = int(sys.argv[2])
    filename1 = sys.argv[3]
    knockout_head1 = int(sys.argv[4])

    with open(filename0, 'rb') as f:
        cfg, state = restricted_unpickle(f)
        model0 = HookedTransformer(cfg)
        model0.load_state_dict(state)

    with open(filename1, 'rb') as f:
        cfg, state = restricted_unpickle(f)
        model1 = HookedTransformer(cfg)
        model1.load_state_dict(state)

    def my_hook_attn_z0(activation, hook):
        a, b, _, c = activation.shape
        activation[:, :, knockout_head0, :] = torch.zeros((a,b,c))
        return activation

    def my_hook_attn_z1(activation, hook):
        a, b, _, c = activation.shape
        activation[:, :, knockout_head1, :] = torch.zeros((a,b,c))
        return activation

    prompts = []
    with open(prompt_filename) as f:
        reader = csv.reader(f)
        for row in reader:
            prompts.append(list(row))

    for prompt in prompts[:num_prompts_to_use]:
        print(prompt)
        tokens = [model0.tokenizer.encode(t)[0] for t in prompt]
        print(tokens)
        ttokens = torch.tensor([tokens])
        logits0 = model0(ttokens).cpu()
        logitsk0 = model0.run_with_hooks(ttokens, fwd_hooks=[('blocks.0.attn.hook_z', my_hook_attn_z0)]).cpu()
        logits1 = model1(ttokens).cpu()
        logitsk1 = model1.run_with_hooks(ttokens, fwd_hooks=[('blocks.0.attn.hook_z', my_hook_attn_z1)]).cpu()
        
        for i in range(len(prompt)):
            fig, ax = plt.subplots(2, 1, squeeze=False)

            plt.title(''.join(prompt[:i+1]))
            ax[0][0].scatter(logits0[0,i,:], logitsk0[0,i,:], s=1)
            ax[1][0].scatter(logits1[0,i,:], logitsk1[0,i,:], s=1)

            values = [(v.item(),j) for j,v in enumerate(logits0[0,i,:])]
            values.sort(reverse=True)

            for _,j in values[:30]:
                ax[0][0].text(logits0[0,i,j].item(), logitsk0[0,i,j].item(), model0.tokenizer.decode(j), fontsize='xx-small')

            values = [(v.item(),j) for j,v in enumerate(logits1[0,i,:])]
            values.sort(reverse=True)

            for _,j in values[:30]:
                ax[1][0].text(logits1[0,i,j].item(), logitsk1[0,i,j].item(), model1.tokenizer.decode(j), fontsize='xx-small')

            plt.show()
