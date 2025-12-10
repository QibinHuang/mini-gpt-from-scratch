"""
mini_gpt_bpe.py
使用 BPE tokenizer + Transformer 训练一个 mini GPT 语言模型

运行前请确保：
1）当前目录下有 input.txt
2）已经运行过 build_tokenizer.py 并生成 tokenizer.json
3）已安装: pip install torch tokenizers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tokenizers import Tokenizer

# ======================
# 0. 配置区域（可以根据机器性能调整）
# ======================
batch_size    = 8        # 每次训练的样本数
block_size    = 16        # 序列长度（context length，单位：token）
max_iters     = 500       # 训练步数：先来800步观察一下
eval_interval = 100       # 每多少步评估/打印一次
learning_rate = 3e-4      # 学习率
eval_iters    = 20        # 每次评估用多少个batch估计loss

device = "cuda" if torch.cuda.is_available() else "cpu"

n_embd = 128              # embedding 维度
n_head = 4                # 注意力头数
n_layer = 3               # transformer block 层数
dropout = 0.1

print("Using device:", device)

# ======================
# 1. 加载文本 & tokenizer & 编码数据
# ======================

# 1.1 读取 input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 1.2 加载 tokenizer.json
tokenizer = Tokenizer.from_file("tokenizer.json")

# 获取特殊token id
bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")
pad_id = tokenizer.token_to_id("[PAD]")

vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size}")

def encode_text(text: str):
    """
    使用 BPE tokenizer 将文本编码为 token id 列表。
    注意：我们在训练 tokenizer 时已经设置了 post_processor，
    会自动在序列前后加 [BOS] 和 [EOS]。
    """
    return tokenizer.encode(text).ids

def decode_tokens(ids):
    # 解码时去掉 PAD
    cleaned = [i for i in ids if i != pad_id]
    return tokenizer.decode(cleaned)

# 1.3 把整个文本编码成一个长的 token 序列
data_ids = encode_text(raw_text)
data = torch.tensor(data_ids, dtype=torch.long)

print("Total tokens in dataset:", len(data))

# 1.4 划分 train / val
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str):
    """
    从 train_data / val_data 中采样 batch。
    x: (B, T) 输入token
    y: (B, T) 目标token = x右移一位
    """
    data_source = train_data if split == "train" else val_data

    # 保证足够长
    if len(data_source) <= block_size + 1:
        raise ValueError(
            f"Data too short for block_size={block_size}. "
            f"Please add more text to input.txt or reduce block_size."
        )

    ix = torch.randint(0, len(data_source) - block_size - 1, (batch_size,))
    x = torch.stack([data_source[i : i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1 : i + 1 + block_size] for i in ix])

    x = x.to(device)
    y = y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    """
    在 train / val 上估计平均loss
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
# 2. 模型定义：GPT 架构
# ======================

class CausalSelfAttention(nn.Module):
    """
    因果自注意力（decoder-only，用下三角 mask 限制只能看过去）
    """
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)

        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 下三角mask，用于因果约束
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    """
    两层前馈网络
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
    Transformer block: 自注意力 + 前馈 + 残差 + LayerNorm（pre-norm）
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
    mini GPT 语言模型：输入token序列，预测下一个token
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embed(idx)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_embed(pos)[None, :, :]
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        自回归生成新token：
        idx: (1, T) 初始token序列
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
# 3. 训练循环
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

            # 生成样例
            # 以一个 [BOS] 作为起点
            context = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=50)
            gen_ids = generated[0].tolist()
            txt = decode_tokens(gen_ids)
            print("=== Sample ===")
            print(txt)
            print("==============")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)

        train_losses_log.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 保存模型和训练loss轨迹
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
    print("模型已保存为 mini_gpt_bpe.pt")

if __name__ == "__main__":
    main()