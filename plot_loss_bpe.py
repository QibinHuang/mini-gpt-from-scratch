"""Plot the training loss stored in a Mini GPT checkpoint."""

import argparse
from pathlib import Path

import matplotlib
import torch


def main():
    parser = argparse.ArgumentParser(description="Plot Mini GPT training loss.")
    parser.add_argument("--checkpoint", type=Path, default=Path("mini_gpt_bpe.pt"))
    parser.add_argument("--output", type=Path, help="Save the figure instead of displaying it.")
    args = parser.parse_args()
    if args.output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    train_losses = checkpoint["train_losses_log"]
    print(f"Loaded {len(train_losses)} training steps.")

    fig, ax = plt.subplots(figsize=(8, 4))
    evaluations = checkpoint.get("evaluation_log")
    if evaluations:
        ax.plot(
            range(1, len(train_losses) + 1), train_losses,
            linewidth=0.8, alpha=0.25, label="Training batches",
        )
        steps = [entry["step"] for entry in evaluations]
        ax.plot(steps, [entry["train"] for entry in evaluations], label="Train")
        ax.plot(steps, [entry["val"] for entry in evaluations], label="Validation")
        ax.legend()
    else:
        ax.plot(range(1, len(train_losses) + 1), train_losses, linewidth=1.5)
    ax.set(xlabel="Training Step", ylabel="Cross-Entropy Loss", title="Mini GPT (BPE) — Loss by Step")
    ax.grid(True)
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"Plot saved to {args.output}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
