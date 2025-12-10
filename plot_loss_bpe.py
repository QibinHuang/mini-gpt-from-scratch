import torch
import matplotlib.pyplot as plt

# 1. 读取训练好的 checkpoint
ckpt_path = "mini_gpt_bpe.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

if "train_losses_log" not in ckpt:
    raise KeyError("train_losses_log not found in checkpoint. "
                   "Make sure you trained with the latest mini_gpt_bpe.py.")

train_losses = ckpt["train_losses_log"]
print(f"Loaded {len(train_losses)} training steps.")

# 2. 构造 x 轴（step 编号）
steps = list(range(len(train_losses)))

# 3. 画图
plt.figure(figsize=(8, 4))
plt.plot(steps, train_losses)
plt.xlabel("Training step")
plt.ylabel("Train loss")
plt.title("Mini GPT (BPE) Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()