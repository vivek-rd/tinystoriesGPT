import torch
from model import GPT, GPTConfig
from transformers import AutoTokenizer

device = 'cuda'

checkpoint = torch.load('./output/ckpt.cpt', map_location=device, weights_only=True)
state_dict = checkpoint['model']
model_args = checkpoint['model_args']
gpt_config = GPTConfig(**model_args)
model = GPT(gpt_config)

unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model = model.to(device)

model.eval()
prompt = "Once upon a time"
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# tokenized_input = tokenizer(prompt, truncation=True, padding='max_length', max_length=block_size)
tokenized_input = (torch.tensor(tokenizer.encode(prompt),
                                dtype=torch.long,
                                device=device)[None, ...])

output = model.generate(tokenized_input, max_new_tokens=200, temperature=1, top_k=20)
decoded_output = [tokenizer.decode(i) for i in output]
print(decoded_output)