import torch
import os
import numpy as np
from model import GPTConfig, GPT

# Paths
data_dir = 'data/openwebtext'
checkpoint_path = 'out/ckpt.pt'
train_bin_path = os.path.join(data_dir, 'val.bin')

# Load train.bin
with open(train_bin_path, 'rb') as f:
    train_data = np.frombuffer(f.read(), dtype=np.uint16)

# Sample first 100,000 tokens
sample_size = 100_000
train_data = train_data[:sample_size]

# Load model
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint['model'])
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Config
block_size = model.config.block_size
stride = block_size  # non-overlapping windows

# Evaluate
def compute_perplexity(data, model, block_size):
    losses = []
    with torch.no_grad():
        for i in range(0, len(data) - block_size, stride):
            x = torch.tensor(data[i:i+block_size], dtype=torch.long)[None, :].to(device)
            y = torch.tensor(data[i+1:i+1+block_size], dtype=torch.long)[None, :].to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    return np.exp(np.mean(losses))  # Perplexity = exp(loss)

# Run
perplexity = compute_perplexity(train_data, model, block_size)
print(f"Perplexity on first {sample_size:,} tokens: {perplexity:.2f}")