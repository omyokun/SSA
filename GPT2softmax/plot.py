import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer

# 1. Load tokenized data
tokens = np.memmap('data/openwebtext/train.bin', dtype=np.uint16, mode='r')

# 2. Load GPT-2 model and tokenizer
model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
embedding_matrix = model.get_input_embeddings().weight.detach().cpu().numpy()  # shape: (vocab_size, emb_dim)

# 3. Compute L2 norm for each token embedding
embedding_norms = np.linalg.norm(embedding_matrix, axis=1)  # shape: (vocab_size,)

# 4. Count token occurrences
token_counts = np.bincount(tokens, minlength=embedding_matrix.shape[0])

# 5. Plot
plt.figure(figsize=(10,6))
plt.scatter(embedding_norms, token_counts, alpha=0.5, s=10)
plt.yscale('log')
plt.xlabel('Token Embedding L2 Norm')
plt.ylabel('Token Count (log scale)')
plt.title('Token Embedding Norm vs. Token Count')
plt.grid(True)
plt.tight_layout()
plt.show()