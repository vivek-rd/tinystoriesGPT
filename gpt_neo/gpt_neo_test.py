import torch
import inspect
from datasets import load_from_disk
from torch.nn.functional import cross_entropy
from transformers import GPTNeoConfig, GPTNeoModel, GPT2Tokenizer, GPTNeoForCausalLM

n_embd = 128
n_layers = 8
n_heads = 8
batch_size = 64
device = 'cuda'

config = GPTNeoConfig(hidden_size=n_embd, num_layers=n_layers, num_heads=n_heads, attention_types=[[["global"], n_layers]])
model = GPTNeoForCausalLM(config)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total number of trainable parameters - {pytorch_total_params}')

test_input = torch.randint(0, 23, size=(10, 23))
test_output = torch.randint(0, 23, size=(10, 23), dtype=torch.int64)
attention_mask = torch.randint(0, 2, size=(10, 23))
output = model(test_input, attention_mask=attention_mask)
print(output.logits.shape)

# print(cross_entropy(input=output.last_hidden_state, target=test_output))

# print(output.last_hidden_state.shape)
# print(output.hidden_states)

print(model.generate(test_input, max_new_tokens=23))