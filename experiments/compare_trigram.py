"""Compare the released model with an interpolated trigram on the existing evaluation splits."""
from collections import Counter
import json
import math
from pathlib import Path

import torch

from evaluate_bpe import baseline_losses, metrics, model_loss
from mini_gpt_bpe import GPTLanguageModel, load_tokenizer


class Trigram:
    def __init__(self, tokens, vocabulary_size):
        self.unigrams = Counter(tokens)
        self.total = len(tokens) + 0.1 * vocabulary_size
        self.counts = {}
        self.totals = Counter()
        for i in range(1, len(tokens)):
            for length in range(1, min(i, 2) + 1):
                context = tuple(tokens[i-length:i])
                if context not in self.counts:
                    self.counts[context] = Counter()
                self.counts[context][tokens[i]] += 1
                self.totals[context] += 1

    def probability(self, context, token):
        probability = (self.unigrams[token] + 0.1) / self.total
        for length in range(1, len(context) + 1):
            suffix = context[-length:]
            counts = self.counts.get(suffix)
            if counts:
                types = len(counts)
                probability = (counts[token] + types * probability) / (self.totals[suffix] + types)
        return probability


def main():
    data = Path("artifacts/shakespeare-eval")
    ckpt = torch.load("demo/model.pt", weights_only=True, map_location="cpu")
    tokenizer = load_tokenizer("demo/tokenizer.json", ckpt["tokenizer_sha256"])
    def encode(path):
        return tokenizer.encode(path.read_text(), add_special_tokens=False).ids

    train = encode(data / "train.txt")
    size = tokenizer.get_vocab_size()
    baseline = Trigram(train, size)
    model = GPTLanguageModel(**ckpt["config"]).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    results = {}
    for name, path in [("plays", data / "test.txt"), ("sonnets", Path("artifacts/window-study/sonnets.txt"))]:
        ids = encode(path)
        losses = baseline_losses(train, ids, size)
        losses["interpolated_trigram"] = -sum(
            math.log(baseline.probability(tuple(ids[max(0, i-2):i]), ids[i]))
            for i in range(1, len(ids))
        ) / (len(ids) - 1)
        losses["mini_gpt"] = model_loss(model, torch.tensor(ids))
        results[name] = {"targets": len(ids)-1, "scores": {k: metrics(v) for k, v in losses.items()}}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
