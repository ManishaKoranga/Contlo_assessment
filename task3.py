'''Finally, create a training loop considering these following requirements:

1. **Single GPU Training Loop:** Your base implementation should be equipped to train your model on a single GPU setup.
2. **Distributed Data Parallel (DDP):** Extend your single GPU training loop to support training across multiple GPUs using DDP. Revisit the [PyTorch's DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) for guidance.
3. **Fully Sharded Data Parallel (FSDP):** Implement FSDP as a part of your training loop to shard the model parameters, gradients, and optimizer state. You can follow [Gupta et al., 2020, Training GPT-3 Like Models on a Single Machine](https://arxiv.org/pdf/2101.06840.pdf) for a comprehensive understanding of it.

**Deliverable:** A Python script containing a functional training loop that is compatible with single GPU, DDP, and FSDP options along with a documentation illustrating how the code adapts to each setting.'''

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed import init_process_group
# hyperparameters
batch_size = 16 
block_size = 32 
max_iters = 100
eval_interval = 5000
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 464
n_head = 16
n_layer = 48
dropout = 0.1


torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])  


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)

            # If using DataParallel, average the loss across replicas
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()

            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
        
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(n_head, n_embd // n_head)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer):
        super(GPT2, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx




save_path = 'gpt2_task3.pth'
fsdp_flag=True #true for fsdp false for ddp




if os.path.isfile(save_path):
    # Load the existing model
    model = GPT2(vocab_size, n_embd, n_head, n_layer)
    model.load_state_dict(torch.load(save_path))
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {save_path}")
    print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
else:
    # Train a new model
    model = GPT2(vocab_size, n_embd, n_head, n_layer)
    model = model.to(device)
    
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1: #DDP
        print(f'{torch.cuda.device_count()} GPUs available')
        model = nn.DataParallel(model)
    elif fsdp_flag == True:  # FSDP
        if device == 'cuda':
            torch.cuda.set_device(0)  
            os.environ['MASTER_ADDR'] = 'localhost'  
            os.environ['MASTER_PORT'] = '12355'  
            os.environ['RANK'] = '0' 
            os.environ['WORLD_SIZE'] = '1'  
            init_process_group(backend='nccl', init_method='env://')
        model = FullyShardedDataParallel(model)
    else:   # single gpu
        model = model

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for iter in range(max_iters):
        print(f'Iteration {iter}')
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train')

        # Adjust target shape for DataParallel
        if isinstance(model, nn.DataParallel):
            yb = yb.view(-1)

        logits, loss = model(xb, yb)
    
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the final model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Use the model for generation
model.eval()  # Set the model to evaluation model

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
print(generated_text)
