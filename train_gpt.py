import os
import torch

import matplotlib.pyplot as plt
from model import GPTConfig, GPT
from datasets import load_from_disk
from transformers import AutoTokenizer

import torch._dynamo
torch._dynamo.config.suppress_errors = True

NUM_EPOCHS = 1
device = 'cuda'

vocab_size = 5000
n_layer = 6
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

checkpoint_iter = 1000 # save checkpoint after every 1000 iterations

init_from = 'resume'
resume_checkpoint_path = './output/ckpt_batch_num_1051.cpt'

torch.set_float32_matmul_precision("high")

dataset = load_from_disk('train_tiny_tokenizer_tokenized_tinystories')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

val_dataset = load_from_disk('validation_tiny_tokenizer_tokenized_tinystories')
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

if init_from == 'resume' and (not os.path.exists(resume_checkpoint_path)):
    print('Changing the init_from variable to scratch i.e training the model from scratch as the checkpoint does not exist')
    init_from = 'scratch'

if init_from == 'resume' and os.path.exists(resume_checkpoint_path):
    checkpoint = torch.load(resume_checkpoint_path, map_location=device)
    state_dict = checkpoint['model']
    model_args = checkpoint['model_args']

    # import losses values for plotting the train_vs_eval curve
    batch_losses = checkpoint['batch_losses']
    eval_losses = checkpoint['eval_losses']
    eval_loss_index = checkpoint['eval_loss_index']
    global_batch_index = checkpoint['global_batch_index']
    print(f'Length of batch losses - {len(batch_losses)}')

    gpt_config = GPTConfig(**model_args)
    model = GPT(gpt_config)

    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
else:
    print('Training the model from scratch')
    model_args = {'vocab_size':vocab_size, 'n_layer':n_layer, 'n_head':n_head, 'n_embd':n_embd, 'block_size':block_size, 'bias':bias, 'dropout':dropout}
    gpt_config = GPTConfig(**model_args)
    model = GPT(gpt_config)

    batch_losses = []
    eval_losses = []
    eval_loss_index = []
    global_batch_index = 0

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


def save_model(
    model, 
    optimizer, 
    model_args,
    batch_losses,
    eval_losses,
    eval_loss_index,
    global_batch_index,
    checkpoint_path
    ):

    checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'batch_losses': batch_losses,
    'eval_losses': eval_losses,
    'eval_loss_index': eval_loss_index,
    'global_batch_index': global_batch_index
    }

    torch.save(checkpoint, os.path.join(checkpoint_path, f'ckpt_batch_num_{global_batch_index}.cpt'))


def plot_training_curve(batch_losses, eval_losses, eval_loss_index, global_batch_index):
    plt.plot(batch_losses, label='training loss')
    plt.scatter(eval_loss_index, eval_losses, color='orange', label='validation loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'train_vs_eval_{global_batch_index}.png')


for epoch in range(NUM_EPOCHS):
    
    model.train()
    running_loss = 0
    average_batch_loss = 0
    
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
        batch_losses.append(loss.item())

        if average_batch_loss < 2.2 and average_batch_loss > 0.9 :
            print(f'Final stopping loss - {average_batch_loss}')
            break

        if idx % 50 == 0:
            # average batch loss for every 50 batches
            average_batch_loss = running_loss / 50

            if idx % 250 == 0:
                eval_loss = calculate_validation_loss(model, val_dataloader)
                eval_loss_index.append(global_batch_index)
                eval_losses.append(eval_loss)
                print(f"epoch - {epoch} | batch - {idx} | average_batch_loss - {average_batch_loss:.4f} | val_loss - {eval_loss:.4f}")
            else:
                print(f"epoch - {epoch} | batch - {idx} | average_batch_loss - {average_batch_loss:.4f}")
            running_loss = 0
        
        if global_batch_index % checkpoint_iter == 0 and global_batch_index != 0:
            save_model(model,
                    optimizer,
                    model_args,
                    batch_losses,
                    eval_losses,
                    eval_loss_index,
                    global_batch_index,
                    checkpoint_path='./checkpoints')

        global_batch_index += 1



# checkpoint and save the model
# write logic to check if init_from is resume to resume training from a checkpoint
# write a function to evaluate validation loss
# write or reuse the python file to generate samples

# use rope embeddings
# checkpoint for every x iterations
# generate a plot of training loss 
# switch to a learning iteration scheduler?
# reduce vocab size


save_model(model,
            optimizer,
            model_args,
            batch_losses,
            eval_losses,
            eval_loss_index,
            global_batch_index,
            checkpoint_path='./output')

prompt = 'Once upon a time'
tokenizer = AutoTokenizer.from_pretrained('tiny_tokenizer')
tokenizer.pad_token = tokenizer.eos_token

# tokenized_input = tokenizer(prompt, truncation=True, padding='max_length', max_length=block_size)
tokenized_input = (torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, ...])

model.eval()
output = model.generate(tokenized_input, max_new_tokens=50, temperature=1, top_k=20)
decoded_output = [tokenizer.decode(i) for i in output]
print(decoded_output)

plot_training_curve(batch_losses, eval_losses, eval_loss_index, global_batch_index)