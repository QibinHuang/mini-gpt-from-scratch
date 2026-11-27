"""Generate text from a saved Mini GPT checkpoint and its matching tokenizer."""

import argparse
import math
from pathlib import Path

import torch

from mini_gpt_bpe import GPTLanguageModel, load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with Mini GPT.")
    parser.add_argument("--checkpoint", type=Path, default=Path("mini_gpt_bpe.pt"))
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--top-k", type=int, help="Sample among the K most likely tokens.")
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")

    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and non-negative")

    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = load_tokenizer(args.tokenizer, checkpoint.get("tokenizer_sha256"))
    config = checkpoint["config"]
    if tokenizer.get_vocab_size() != config["vocab_size"]:
        parser.error("the tokenizer vocabulary size does not match the checkpoint")
    if args.top_k is not None and not 1 <= args.top_k <= config["vocab_size"]:
        parser.error(f"--top-k must be between 1 and {config['vocab_size']}")
    bos_id = tokenizer.token_to_id("[BOS]")
    if bos_id is None:
        parser.error("the tokenizer must contain [BOS]")

    model = GPTLanguageModel(**config).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    context = torch.tensor([[bos_id, *prompt_ids]], dtype=torch.long)
    generated = model.generate(
        context, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
