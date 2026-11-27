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

import hashlib
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tokenizers import Tokenizer, decoders

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

# ======================
# 1. Batch sampling and loss estimation
# ======================

def get_batch(data_source, context_length, generator=None, batch_count=batch_size):
    """
    Sample a batch of sequences from the dataset.

    Returns:
        x: (B, T) input tokens
        y: (B, T) target tokens (shifted by one position)
    """
    # Ensure dataset is long enough
    if len(data_source) <= context_length:
        raise ValueError(
            f"Data too short for context_length={context_length}. "
            f"Add more text to input.txt or reduce context_length."
        )

    ix = torch.randint(0, len(data_source) - context_length, (batch_count,), generator=generator)
    x = torch.stack([data_source[i : i + context_length] for i in ix])
    y = torch.stack([data_source[i + 1 : i + 1 + context_length] for i in ix])

    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, evaluation_seed):
    """
    Compute average train/val loss over eval_iters batches.
    """
    out = {}
    generator = torch.Generator().manual_seed(evaluation_seed)
    model.eval()

    for split, data_source in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data_source, model.context_length, generator)
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
    def __init__(self, n_embd, n_head, dropout, context_length):
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
        mask = torch.tril(torch.ones(context_length, context_length))[None, None, :, :]
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
    def __init__(self, n_embd, n_head, dropout, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, context_length)
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
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout, context_length=block_size):
        super().__init__()

        self.context_length = context_length

        # Token & positional embeddings
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed   = nn.Embedding(context_length, n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout, context_length) for _ in range(n_layer)]
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

    def forward(self, idx, targets=None, *, last_only=False):
        """
        Forward pass.
        idx: (B, T) token IDs
        targets: (B, T) token IDs for supervised training
        """
        if last_only and targets is not None:
            raise ValueError("last_only is for inference without training targets")
        B, T = idx.shape

        # Token + positional embeddings
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(torch.arange(T, device=idx.device))[None, :, :]
        x = tok_emb + pos_emb

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x[:, -1:] if last_only else x)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape(B * T, logits.size(-1))
            targets_flat = targets.reshape(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, *, temperature=1.0, top_k=None):
        """Sample new tokens, or use greedy decoding when temperature is zero."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond, last_only=True)
            scores = logits[:, -1, :]

            if temperature == 0:
                next_id = scores.argmax(dim=-1, keepdim=True)
            else:
                candidates = None
                if top_k is not None:
                    scores, candidates = scores.topk(top_k, dim=-1)
                if temperature != 1:
                    centered = scores.double() - scores.max(dim=-1, keepdim=True).values
                    scores = centered / temperature
                probs = F.softmax(scores, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                if candidates is not None:
                    next_id = candidates.gather(-1, next_id)

            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ======================
# 3. Training loop
# ======================

def load_tokenizer(path, expected_sha256=None):
    if expected_sha256 and hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Tokenizer file does not match the checkpoint fingerprint")
    tokenizer = Tokenizer.from_file(str(path))
    # Older tokenizers in this repo omit the decoder.
    if tokenizer.decoder is None:
        tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Mini GPT on a text corpus.")
    parser.add_argument("--input", type=Path, default=Path("input.txt"))
    parser.add_argument("--validation", type=Path, help="Use a separate validation text file.")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--output", type=Path, default=Path("mini_gpt_bpe.pt"))
    parser.add_argument("--best-output", type=Path, help="Save the weights with the lowest validation loss.")
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--steps", type=int, default=max_iters + 1)
    parser.add_argument("--context-length", type=int, default=block_size)
    parser.add_argument("--eval-interval", type=int, default=eval_interval)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.context_length, args.eval_interval) < 1:
        parser.error("--steps, --context-length, and --eval-interval must be positive")
    if args.best_output and args.best_output.resolve() == args.output.resolve():
        parser.error("--output and --best-output must use different paths")
    if args.validation and args.validation.resolve() == args.input.resolve():
        parser.error("--input and --validation must use different files")

    torch.manual_seed(args.seed)
    print("Using device:", device)
    tokenizer = load_tokenizer(args.tokenizer)
    bos_id = tokenizer.token_to_id("[BOS]")
    if bos_id is None:
        parser.error("the tokenizer must contain [BOS]")
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size}")

    data = torch.tensor(tokenizer.encode(args.input.read_text(encoding="utf-8")).ids, dtype=torch.long)
    if args.validation:
        validation_ids = tokenizer.encode(args.validation.read_text(encoding="utf-8")).ids
        train_data, val_data = data, torch.tensor(validation_ids, dtype=torch.long)
    else:
        split = int(0.9 * len(data))
        train_data, val_data = data[:split], data[split:]
    if min(len(train_data), len(val_data)) <= args.context_length:
        parser.error("each split must contain more tokens than --context-length")
    print(f"Train tokens: {len(train_data):,}. Validation tokens: {len(val_data):,}.")

    config = {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "context_length": args.context_length,
    }
    model = GPTLanguageModel(**config).to(device)
    print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_losses_log = []
    evaluation_log = []
    best_loss = float("inf")
    best_weights = None
    best_step = None

    for step in range(args.steps + 1):
        if step % args.eval_interval == 0 or step == args.steps:
            losses = estimate_loss(model, train_data, val_data, args.seed)
            evaluation_log.append({"step": step, **losses})
            print(f"Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if args.best_output and step > 0 and losses["val"] < best_loss:
                best_loss = losses["val"]
                best_step = step
                best_weights = deepcopy(model.state_dict())

            model.eval()
            context = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=50)
            print("=== Sample ===")
            print(tokenizer.decode(generated[0].tolist()))
            print("==============")
            model.train()

        if step == args.steps:
            break
        xb, yb = get_batch(train_data, args.context_length, batch_count=args.batch_size)
        _, loss = model(xb, yb)
        train_losses_log.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    checkpoint = {
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "config": config,
        "train_losses_log": train_losses_log,
        "evaluation_log": evaluation_log,
        "model_state_dict": model.state_dict(),
        "model_step": args.steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Final checkpoint saved to {args.output}")
    if args.best_output:
        args.best_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {**checkpoint, "model_state_dict": best_weights, "model_step": best_step},
            args.best_output,
        )
        print(f"Best checkpoint saved to {args.best_output} (step {best_step}, val loss {best_loss:.4f})")


if __name__ == "__main__":
    main()
