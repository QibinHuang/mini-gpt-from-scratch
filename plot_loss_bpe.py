"""
plot_loss_bpe.py
Load a saved training checkpoint (mini_gpt_bpe.pt)
and visualize the training loss curve.

This script is used to inspect model convergence behavior.
"""

import torch
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Load the trained checkpoint
# ---------------------------------------
ckpt_path = "mini_gpt_bpe.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")

# Ensure the checkpoint contains the training loss log
if "train_losses_log" not in ckpt:
    raise KeyError(
        "train_losses_log not found in checkpoint. "
        "Make sure you trained with the latest mini_gpt_bpe.py."
    )

train_losses = ckpt["train_losses_log"]
print(f"Loaded {len(train_losses)} training steps.")

# ---------------------------------------
# 2. Build the x-axis (training steps)
# ---------------------------------------
steps = list(range(len(train_losses)))

# ---------------------------------------
# 3. Plot the training loss curve
# ---------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(steps, train_losses, linewidth=1.5)

plt.xlabel("Training Step")
plt.ylabel("Training Loss")
plt.title("Mini GPT (BPE) — Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()
