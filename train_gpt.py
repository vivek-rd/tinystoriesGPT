import os
import torch

from model import GPTConfig, GPT
from datasets import load_from_disk
from transformers import AutoTokenizer

import torch._dynamo
torch._dynamo.config.suppress_errors = True

NUM_EPOCHS = 1
device = 'cuda'

vocab_size = 50304
n_layer = 8
n_head = 8
n_embd = 128
block_size = 256
batch_size = 64

bias = False
dropout = 0.0

# adamw optimizer
learning_rate = 6e-3 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = False
compile = True

init_from = 'resume'

torch.set_float32_matmul_precision("high")

dataset = load_from_disk('tokenized_tinystories')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

val_dataset = load_from_disk('validation_tokenized_tinystories')
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

model_args = {'vocab_size':vocab_size, 'n_layer':n_layer, 'n_head':n_head, 'n_embd':n_embd, 'block_size':block_size, 'bias':bias, 'dropout':dropout}

if init_from == 'resume' and os.path.exists('./output/ckpt.cpt'):
    checkpoint = torch.load('./output/ckpt.cpt', map_location=device)
    state_dict = checkpoint['model']
    model_args = checkpoint['model_args']
    gpt_config = GPTConfig(**model_args)
    model = GPT(gpt_config)

    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
else:
    gpt_config = GPTConfig(**model_args)
    model = GPT(gpt_config)

model = model.to(device)
if compile:
    model = torch.compile(model)
print(f"Number of trainable parameters - {model.get_num_params}")

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory
scaler = torch.amp.GradScaler('cuda')


def calculate_validation_loss(model, val_dataloader):
    model.eval()
    eval_loss = 0.0

    for idx, batch in enumerate(val_dataloader):
        X, Y = batch['input_ids'][:, 0:-1], batch['input_ids'][:, 1:]
        X, Y = X.to(device), Y.to(device)
        attention_mask = batch['attention_mask'][:, 0:-1]

        logits, loss = model(X, attention_mask=attention_mask, targets=Y)
        eval_loss += loss.item()
    
    model.train()
    return eval_loss/len(val_dataloader)


# def plot_training_curve()


for epoch in range(NUM_EPOCHS):
    
    model.train()
    running_loss = 0
    batch_loss = 0
    
    for idx, batch in enumerate(dataloader):
        X, Y = batch['input_ids'][:, 0:-1], batch['input_ids'][:, 1:]
        X, Y = X.to(device), Y.to(device)
        attention_mask = batch['attention_mask'][:, 0:-1]

        with torch.autocast(device_type=device, dtype=torch.float16):
            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(X, attention_mask=attention_mask, targets=Y)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

        running_loss += loss.item()

        if batch_loss < 2.05 and batch_loss > 0.9 :
            print(f'Final stopping loss - {batch_loss}')
            break

        if idx % 50 == 0:
            # average batch loss for every 1000 batches
            batch_loss = running_loss / 50
            if idx % 250 == 0:
                eval_loss = calculate_validation_loss(model, val_dataloader)
                print(f"epoch - {epoch} | batch - {idx} | batch_loss - {batch_loss:.4f} | val_loss - {eval_loss:.4f}")
            else:
                print(f"epoch - {epoch} | batch - {idx} | batch_loss - {batch_loss:.4f}")
            running_loss = 0

# checkpoint and save the model
# write logic to check if init_from is resume to resume training from a checkpoint
# write a function to evaluate validation loss
# write or reuse the python file to generate samples

# use rope embeddings
# checkpoint for every x iterations
# generate a plot of training loss 
# switch to a learning iteration scheduler?
# reduce vocab size


checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args
}
torch.save(checkpoint, './output/ckpt.cpt')

prompt = 'Once upon a time'
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# tokenized_input = tokenizer(prompt, truncation=True, padding='max_length', max_length=block_size)
tokenized_input = (torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, ...])

model.eval()
output = model.generate(tokenized_input, max_new_tokens=50, temperature=1, top_k=20)
print(type(output))
print(output)
decoded_output = [tokenizer.decode(i) for i in output]
print(decoded_output)

