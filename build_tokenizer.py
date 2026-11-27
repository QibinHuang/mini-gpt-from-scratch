"""
build_tokenizer.py
Train a Byte Pair Encoding (BPE) tokenizer from input.txt
and save it as tokenizer.json.

This tokenizer will be used by the mini GPT model.
"""

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

def main():
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument("--input", type=Path, default=Path("input.txt"))
    parser.add_argument("--output", type=Path, default=Path("tokenizer.json"))
    args = parser.parse_args()

    # ---------------------------------------
    # 1. Initialize an empty BPE tokenizer
    # ---------------------------------------
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Use ByteLevel pre-tokenization (commonly used for English text)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    # ---------------------------------------
    # 2. Configure the BPE trainer
    # ---------------------------------------
    vocab_size = 2000  # Sufficient for small datasets; increase for larger corpora
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # ---------------------------------------
    # 3. Train the tokenizer on input.txt
    # ---------------------------------------
    tokenizer.train(files=[str(args.input)], trainer=trainer)

    # ---------------------------------------
    # 4. Add BOS/EOS template post-processing
    # Automatically wraps sequences like:
    #   [BOS] sentence tokens [EOS]
    # ---------------------------------------
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", bos_id),
            ("[EOS]", eos_id),
        ],
    )

    # ---------------------------------------
    # 5. Save the tokenizer to file
    # ---------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output))

    print(f"Tokenizer saved to {args.output}")
    print("Vocab size:", tokenizer.get_vocab_size())


if __name__ == "__main__":
    main()
