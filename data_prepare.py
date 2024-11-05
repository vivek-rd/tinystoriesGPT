import torch
from datasets import load_dataset
from transformers import AutoTokenizer

BLOCK_SIZE = 256

def create_dataset(tokenizer_model, block_size, split='train'):

    dataset = load_dataset('roneneldan/TinyStories', split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    
    # gpt2 does not have padding token, so using eos as padding token
    if tokenizer_model == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(lambda e: tokenizer(e['text'], 
                                            truncation=True, 
                                            padding='max_length', 
                                            # max_length plus one to get the next tokens
                                            max_length=block_size + 1
                                            ), batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataset.save_to_disk(f'{split}_tokenized_tinystories')

create_dataset('gpt2', block_size=BLOCK_SIZE, split='validation')