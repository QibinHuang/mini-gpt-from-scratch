"""
mini_gpt_bpe.py
Train a small GPT-style language model using:
- A BPE tokenizer (trained separately)
- A decoder-only Transformer architecture

Requirements:
1) An input text file named input.txt in the same directory
2) A tokenizer.json file generated via build_tokenizer.py
3) Install dependencies: pip install torch tokenizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tokenizers import Tokenizer

# ======================
# 0. Configuration (adjust based on hardware)
# ======================
batch_size    = 8        # Batch size
block_size    = 16       # Context length (number of tokens)
max_iters     = 500      # Number of training iterations
eval_interval = 100      # Print losses every N steps
learning_rate = 3e-4     # Learning rate
eval_iters    = 20       # Number of batches for loss estimation

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model sizes
n_embd = 128             # Embedding dimension
n_head = 4               # Number of attention heads
n_layer = 3              # Number of Transformer blocks
dropout = 0.1

print("Using device:", device)

# ======================
# 1. Load text, tokenizer, and encode dataset
# ======================

# 1.1 Load raw input text
with open("input.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 1.2 Load tokenizer.json
tokenizer = Tokenizer.from_file("tokenizer.json")

# Retrieve special token IDs
bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")
pad_id = tokenizer.token_to_id("[PAD]")

vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")

def encode_text(text: str):
    """
    Encode text into token IDs using the BPE tokenizer.
    Note: During tokenizer training we already enabled a post-processor,
    so [BOS] and [EOS] are automatically appended.
    """
    return tokenizer.encode(text).ids

def decode_tokens(ids):
    """
    Decode a sequence of token IDs back into text.
    PAD tokens are removed before decoding.
    """
    cleaned = [i for i in ids if i != pad_id]
    return tokenizer.decode(cleaned)

# 1.3 Encode entire dataset into a single long token sequence
data_ids = encode_text(raw_text)
data = torch.tensor(data_ids, dtype=torch.long)

print("Total tokens in dataset:", len(data))

# 1.4 Split into train / val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str):
    """
    Sample a batch of sequences from the dataset.

    Returns:
        x: (B, T) input tokens
        y: (B, T) target tokens (shifted by one position)
    """
    data_source = train_data if split == "train" else val_data

    # Ensure dataset is long enough
    if len(data_source) <= block_size + 1:
        raise ValueError(
            f"Data too short for block_size={block_size}. "
            f"Add more text to input.txt or reduce block_size."
        )

    ix = torch.randint(0, len(data_source) - block_size - 1, (batch_size,))
    x = torch.stack([data_source[i : i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1 : i + 1 + block_size] for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    """
    Compute average train/val loss over eval_iters batches.
    """
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)

    model.train()
    return out

# ======================
# 2. Model Architecture: Decoder-only GPT
# ======================

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention layer (decoder-only).
    Uses a lower-triangular mask so each token can attend only to previous ones.
    """
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0

        self.n_head = n_head
        self.head_dim = n_embd // n_head

        # Linear projections
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular (T, T)
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply causal mask
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    Transformer block: LayerNorm → Self-Attention → LayerNorm → FeedForward,
    each wrapped with residual connections (pre-norm architecture).
    """
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    Mini GPT language model.
    Takes token sequences as input and predicts the next token.
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()

        # Token & positional embeddings
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed   = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights following GPT-style initialization.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        Forward pass.
        idx: (B, T) token IDs
        targets: (B, T) token IDs for supervised training
        """
        B, T = idx.shape

        # Token + positional embeddings
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(torch.arange(T, device=idx.device))[None, :, :]
        x = tok_emb + pos_emb

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, vocab_size)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generate new tokens starting from idx.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits_last = logits[:, -1, :]
            probs = F.softmax(logits_last, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ======================
# 3. Training loop
# ======================

def main():
    model = GPTLanguageModel(vocab_size, n_embd, n_head, n_layer, dropout).to(device)
    print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    train_losses_log = []

    for iter in range(max_iters + 1):
        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"Step {iter}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

            # Generate a sample sequence using [BOS] as the seed
            context = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=50)
            gen_ids = generated[0].tolist()
            print("=== Sample ===")
            print(decode_tokens(gen_ids))
            print("==============")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        train_losses_log.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save model checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,
                "dropout": dropout,
            },
            "train_losses_log": train_losses_log,
        },
        "mini_gpt_bpe.pt",
    )
    print("Model saved as mini_gpt_bpe.pt")

if __name__ == "__main__":
    main()
