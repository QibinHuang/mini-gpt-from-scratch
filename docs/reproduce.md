Mini GPT — Train a Transformer From Scratch (BPE Tokenizer + PyTorch)

The goal is to understand a GPT-style language model by implementing and
measuring the entire next-token prediction pipeline on a laptop. The project
covers BPE tokenization, a decoder-only Transformer with causal attention,
training and checkpoint selection, held-out evaluation, and text generation
from a prompt.

The model is a small teaching implementation. Its results show what this
architecture learns on a particular text corpus at a small scale.

## Run from scratch

The CS336 chat sketches the decoder-only Transformer. This repository implements
the model in PyTorch and adds BPE tokenization, next-token training, text sampling,
and loss plotting.

From the repository root, using Python 3.11:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python build_tokenizer.py --output artifacts/tokenizer.json
python mini_gpt_bpe.py --tokenizer artifacts/tokenizer.json --output artifacts/mini_gpt_bpe.pt
python plot_loss_bpe.py --checkpoint artifacts/mini_gpt_bpe.pt --output artifacts/training-loss.png
```

Training prints train and validation loss every 100 steps and generates sample
text at each evaluation. The bundled corpus is a small example for testing the
pipeline and observing overfitting. Use `--input` with the same corpus path in
the tokenizer and training commands to try another text.

To plot the checkpoint already included in the repository:

```sh
python plot_loss_bpe.py --output artifacts/original-training-loss.png
```

## Train and evaluate on Tiny Shakespeare

The corpus is split by line into 90% training, 5% validation, and 5% test.
BPE is fitted on the training text. Validation uses the same sampled windows
throughout training and determines which checkpoint to keep. The test text
is scored after checkpoint selection.

```sh
mkdir -p artifacts/shakespeare-eval
curl -fL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  -o artifacts/shakespeare-eval/input.txt

python - <<'PY'
from pathlib import Path

directory = Path("artifacts/shakespeare-eval")
lines = (directory / "input.txt").read_text(encoding="utf-8").splitlines(keepends=True)
train_end, validation_end = len(lines) * 90 // 100, len(lines) * 95 // 100
(directory / "train.txt").write_text("".join(lines[:train_end]), encoding="utf-8")
(directory / "validation.txt").write_text("".join(lines[train_end:validation_end]), encoding="utf-8")
(directory / "test.txt").write_text("".join(lines[validation_end:]), encoding="utf-8")
PY

python build_tokenizer.py \
  --input artifacts/shakespeare-eval/train.txt \
  --output artifacts/shakespeare-eval/tokenizer.json

OMP_NUM_THREADS=4 python mini_gpt_bpe.py \
  --input artifacts/shakespeare-eval/train.txt \
  --validation artifacts/shakespeare-eval/validation.txt \
  --tokenizer artifacts/shakespeare-eval/tokenizer.json \
  --context-length 64 --steps 1600 --eval-interval 200 \
  --output artifacts/shakespeare-eval/final.pt --best-output artifacts/shakespeare-eval/best.pt

python plot_loss_bpe.py \
  --checkpoint artifacts/shakespeare-eval/final.pt \
  --output artifacts/shakespeare-eval/loss.png

OMP_NUM_THREADS=4 python evaluate_bpe.py \
  --checkpoint artifacts/shakespeare-eval/best.pt \
  --tokenizer artifacts/shakespeare-eval/tokenizer.json \
  --train artifacts/shakespeare-eval/train.txt \
  --test artifacts/shakespeare-eval/test.txt

python generate_bpe.py \
  --checkpoint artifacts/shakespeare-eval/best.pt \
  --tokenizer artifacts/shakespeare-eval/tokenizer.json \
  --prompt "ROMEO:" --tokens 80
```

The evaluator compares all models on the same next-token targets. The unigram
baseline uses overall token frequencies. The bigram baseline conditions on
the preceding token. Mini GPT uses up to 64 tokens of context and restarts
context at each evaluation window. Lower cross-entropy and perplexity
indicate better next-token predictions on this tokenization and test split.

For a separate corpus, use the tokenizer produced from that run when loading its
weights. The original bundled checkpoint can still be used directly with
`python generate_bpe.py --prompt "Language models"`.

## Reference CPU run

The 64-token, 1,600-step configuration above selected step 1,600 by validation
loss (4.10). The test set contained 19,724 next-token predictions.

| Model | Test cross-entropy | Test perplexity |
| --- | ---: | ---: |
| Unigram | 5.89 | 362.70 |
| Bigram | 4.57 | 96.33 |
| Mini GPT | 4.24 | 69.29 |

The test split was held out from fitting and checkpoint selection in this run.
It overlaps the validation text used during an earlier exploratory run. These
numbers describe this corpus and setup.

## How much context does the model use?

The following diagnostic keeps the weights and the 2,048 test targets fixed.
Each prediction receives only the specified number of preceding tokens.
These scores use sliding context on a sample of targets. The reference
test scores above use every target and restart context at block boundaries.

```sh
OMP_NUM_THREADS=4 python evaluate_bpe.py \
  --checkpoint artifacts/shakespeare-eval/best.pt \
  --tokenizer artifacts/shakespeare-eval/tokenizer.json \
  --train artifacts/shakespeare-eval/train.txt \
  --test artifacts/shakespeare-eval/test.txt \
  --context-sweep 1 4 16 32 64
```

| Preceding tokens | Perplexity on shared targets |
| ---: | ---: |
| 1 | 98.16 |
| 4 | 73.18 |
| 16 | 69.21 |
| 32 | 68.48 |
| 64 | 70.74 |

The largest improvement on this sample came from going from one to four
preceding tokens. This experiment shortens the input to an existing model
trained with a 64-token context. Comparing training context lengths requires
training separate models and selecting settings using validation results.

## Compare decoding settings

Use a single prompt and checkpoint to see how decoding affects the output.
The example samples from the 40 highest-scoring tokens at temperature 0.8.
Use `--temperature 0` for greedy decoding. By default the generator uses
temperature 1 and samples from the full vocabulary.

```sh
python generate_bpe.py \
  --checkpoint artifacts/shakespeare-eval/best.pt \
  --tokenizer artifacts/shakespeare-eval/tokenizer.json \
  --prompt "ROMEO:" --tokens 80 --seed 42 --temperature 0.8 --top-k 40
```

## Reproduce the six-run study and external evaluation

From the repository root, use the Tiny Shakespeare data preparation commands
above. For the window study, copy the released tokenizer so token IDs match
the reference experiment. The demo tokenizer was fitted on the training split.
Each run uses 512 prediction targets per update. Validation scores cover the
entire validation split with each model's native block length. This measures
each available context in use and also changes the frequency of context resets.
The 32-token model has 4,096 fewer learned positional embedding parameters.

```sh
cp demo/tokenizer.json artifacts/shakespeare-eval/tokenizer.json
python -m experiments.train_windows
OMP_NUM_THREADS=4 python -m experiments.summarize_windows
python -m experiments.benchmark

mkdir -p artifacts/window-study
curl -fL https://raw.githubusercontent.com/GITenberg/Shakespeare-s-Sonnets_1041/master/1041.txt -o artifacts/window-study/sonnets-source.txt
python -m experiments.prepare_sonnets
shasum -a 256 -c results/data.sha256
OMP_NUM_THREADS=4 python evaluate_bpe.py --checkpoint demo/model.pt --tokenizer demo/tokenizer.json --train artifacts/shakespeare-eval/train.txt --test artifacts/window-study/sonnets.txt
```

The summarizer writes the selected checkpoint to the demo directory. Select
the lowest three-seed mean validation cross-entropy, then use seed 42 for
the released model. The external corpus uses the verse of the 154 sonnets
with Roman numeral headings and Gutenberg boilerplate removed.

Reference environment: Python 3.11.5 on macOS arm64. The three directly used
packages are pinned in requirements.txt. Use
`python -m pip install -r results/environment-macos-arm64.txt` to reproduce
the clean smoke-test environment including its transitive dependencies.

For the external evaluation, none of the 2,155 extracted verse lines matched
a complete training line after trimming whitespace and folding case. This is
a line-level overlap check. The same author appears in both corpora.
