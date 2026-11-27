"""Compare Mini GPT with token-frequency baselines on held-out text."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from mini_gpt_bpe import GPTLanguageModel, load_tokenizer


@torch.inference_mode()
def model_loss(model, tokens):
    """Score every target token once, restarting context at window boundaries."""
    context_length = model.context_length
    target_count = len(tokens) - 1
    window_count, tail_length = divmod(target_count, context_length)
    total_loss = 0.0

    if window_count:
        windows = tokens.unfold(0, context_length + 1, context_length)
        for batch in windows.split(16):
            _, loss = model(batch[:, :-1], batch[:, 1:])
            total_loss += loss.item() * batch[:, 1:].numel()

    if tail_length:
        tail = tokens[-(tail_length + 1):].unsqueeze(0)
        _, loss = model(tail[:, :-1], tail[:, 1:])
        total_loss += loss.item() * tail_length
    return total_loss / target_count


@torch.inference_mode()
def context_losses(model, tokens, lengths, sample_count):
    """Score each context length on the same evenly spaced target positions."""
    max_length = max(lengths)
    count = min(sample_count, len(tokens) - max_length)
    positions = max_length + torch.arange(count) * (len(tokens) - max_length) // count
    losses = {}
    for length in lengths:
        offsets = torch.arange(-length, 0)
        total_loss = 0.0
        for batch in positions.split(32):
            inputs = tokens[batch[:, None] + offsets]
            logits, _ = model(inputs)
            total_loss += F.cross_entropy(logits[:, -1], tokens[batch], reduction="sum").item()
        losses[length] = total_loss / count
    return count, losses


def metrics(loss):
    return {"cross_entropy": loss, "perplexity": math.exp(loss)}


def baseline_losses(train_tokens, test_tokens, vocab_size):
    """Fit add-0.1 unigram and bigram models using training tokens only."""
    smoothing = 0.1
    train_pairs = Counter(zip(train_tokens[:-1], train_tokens[1:]))
    train_contexts = Counter(train_tokens[:-1])
    train_targets = Counter(train_tokens[1:])
    test_pairs = Counter(zip(test_tokens[:-1], test_tokens[1:]))
    normalizer = len(train_tokens) - 1 + smoothing * vocab_size
    unigram_loss = bigram_loss = 0.0

    for (previous, target), count in test_pairs.items():
        unigram_probability = (train_targets[target] + smoothing) / normalizer
        bigram_probability = (train_pairs[previous, target] + smoothing) / (
            train_contexts[previous] + smoothing * vocab_size
        )
        unigram_loss -= count * math.log(unigram_probability)
        bigram_loss -= count * math.log(bigram_probability)

    target_count = len(test_tokens) - 1
    return {"unigram": unigram_loss / target_count, "bigram": bigram_loss / target_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True, help="Text used to fit the baselines.")
    parser.add_argument("--test", type=Path, required=True, help="Held-out text scored by every model.")
    parser.add_argument("--context-sweep", nargs="+", type=int, metavar="TOKENS",
                        help="Also evaluate these preceding-context lengths on shared targets.")
    parser.add_argument("--context-samples", type=int, default=2048,
                        help="Maximum number of shared targets for the context sweep.")
    args = parser.parse_args()
    if args.train.samefile(args.test):
        parser.error("--train and --test must use different files")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = load_tokenizer(args.tokenizer, checkpoint.get("tokenizer_sha256"))
    config = checkpoint["config"]
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != config["vocab_size"]:
        parser.error("the tokenizer vocabulary size does not match the checkpoint")

    train_tokens = tokenizer.encode(args.train.read_text(encoding="utf-8"), add_special_tokens=False).ids
    test_tokens = tokenizer.encode(args.test.read_text(encoding="utf-8"), add_special_tokens=False).ids
    if min(len(train_tokens), len(test_tokens)) < 2:
        parser.error("each text must contain at least two tokens")

    model = GPTLanguageModel(**config).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    lengths = sorted(set(args.context_sweep or []))
    if lengths and (lengths[0] < 1 or lengths[-1] > model.context_length):
        parser.error(f"context lengths must be between 1 and {model.context_length}")
    if lengths and (args.context_samples < 1 or lengths[-1] >= len(test_tokens)):
        parser.error("the sweep needs a positive sample count and at least one eligible target")

    tokens = torch.tensor(test_tokens, dtype=torch.long)
    losses = baseline_losses(train_tokens, test_tokens, vocab_size)
    losses["mini_gpt"] = model_loss(model, tokens)
    result = {
        "checkpoint_step": checkpoint.get("model_step"),
        "test_predictions": len(test_tokens) - 1,
        "context_length": model.context_length,
        "scores": {name: metrics(loss) for name, loss in losses.items()},
    }
    if lengths:
        count, context_scores = context_losses(model, tokens, lengths, args.context_samples)
        result["context_sweep"] = {
            "shared_targets": count,
            "scores_by_context_length": {
                length: metrics(loss) for length, loss in context_scores.items()
            },
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
