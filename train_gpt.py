import torch
import torch._dynamo
from model import GPTConfig, GPT
from datasets import load_from_disk
torch._dynamo.config.suppress_errors = True

BATCH_SIZE = 64
NUM_EPOCHS = 1
device_type = 'cuda'

vocab_size = 50304
n_layer = 6
n_head = 8
n_embd = 128

# adamw optimizer
learning_rate = 6e-3 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = False
compile = True

torch.set_float32_matmul_precision("high")

dataset = load_from_disk('tokenized_tinystories')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

config_dict = {'vocab_size':vocab_size, 'n_layer':n_layer, 'n_head':n_head, 'n_embd':n_embd}
gpt_config = GPTConfig(**config_dict)
model = GPT(gpt_config)

model = model.to(device_type)
if compile:
    model = torch.compile(model)
print(f"Number of trainable parameters - {model.get_num_params}")

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

for epoch in range(NUM_EPOCHS):
    
    model.train()
    running_loss = 0
    batch_loss = 0
    
    for idx, batch in enumerate(dataloader):
        X, Y = batch['input_ids'][:, 0:-1], batch['input_ids'][:, 1:]
        X, Y = X.to(device_type), Y.to(device_type)
        attention_mask = batch['attention_mask'][:, 0:-1]

        optimizer.zero_grad()
        logits, loss = model(X, attention_mask=attention_mask, targets=Y)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if idx % 50 == 0:
            # average batch loss for every 1000 batches
            batch_loss = running_loss / 50
            print(f"epoch - {epoch} | batch_loss - {batch_loss:.4f}")
            running_loss = 0
    





        

