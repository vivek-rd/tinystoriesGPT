import torch
import inspect
from datasets import load_from_disk
from torch.nn.functional import cross_entropy
from transformers import GPTNeoConfig, GPTNeoModel, GPT2Tokenizer, GPTNeoForCausalLM

NUM_EPOCHS = 1

n_embd = 128
n_layers = 8
n_heads = 8
batch_size = 64
device = 'cuda'

# adamw optimizer
learning_rate = 6e-3 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
decay_lr = False
compile = True

config = GPTNeoConfig(hidden_size=n_embd, num_layers=n_layers, num_heads=n_heads, attention_types=[[["global"], n_layers]])
model = GPTNeoForCausalLM(config)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'Total number of trainable parameters - {pytorch_total_params}')
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# vocab size
vocab_size = tokenizer.vocab_size

torch.set_float32_matmul_precision("high")

dataset = load_from_disk('tokenized_tinystories')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

val_dataset = load_from_disk('validation_tokenized_tinystories')
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")

    return optimizer

optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device)
loss_fn = cross_entropy()

def calculate_validation_loss(model, val_dataloader):
    model.eval()
    eval_loss = 0.0

    for idx, batch in enumerate(val_dataloader):
        X, Y = batch['input_ids'][:, 0:-1], batch['input_ids'][:, 1:]
        X, Y = X.to(device), Y.to(device)
        attention_mask = batch['attention_mask'][:, 0:-1]

        logits = model(X, attention_mask=attention_mask, targets=Y).last_hidden_state
        loss = cross_entropy(logits.view(-1, vocab_size).to(device), Y.view(-1).to(device))
        eval_loss += loss.item()
    
    model.train()
    return eval_loss/len(val_dataloader)

scaler = torch.amp.GradScaler('cuda')

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
            logits = model(X, attention_mask=attention_mask, targets=Y).last_hidden_state
            loss = cross_entropy(logits.view(-1, vocab_size).to(device), Y.view(-1).to(device))
            
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

prompt = "Once upon a time"
tokenizer.pad_token = tokenizer.eos_token

tokenized_input = (torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)[None, ...])
model.eval()
output = model.generate(tokenized_input, max_new_tokens=50, temperature=1, top_k=20)
print(type(output))
print(output)
decoded_output = [tokenizer.decode(i) for i in output]
print(decoded_output)