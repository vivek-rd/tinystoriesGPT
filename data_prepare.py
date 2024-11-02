import torch
from datasets import load_dataset
from transformers import AutoTokenizer

BATCH_SIZE = 512
BLOCK_SIZE = 256

dataset = load_dataset('roneneldan/TinyStories', split='train')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

dataset = dataset.map(lambda e: tokenizer(e['text'], 
                                          truncation=True, 
                                          padding='max_length', 
                                          # max_length plus one to get the next tokens
                                          max_length=BLOCK_SIZE + 1
                                          ), batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataset.save_to_disk('tokenized_tinystories')

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
print(next(iter(dataloader)))