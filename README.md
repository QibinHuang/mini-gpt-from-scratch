Mini GPT — Train a Transformer From Scratch (BPE Tokenizer + PyTorch)

This project implements a minimal GPT-style language model from scratch using:
	•	Byte Pair Encoding (BPE) tokenizer
	•	Decoder-only Transformer
	•	Autoregressive next-token prediction
	•	PyTorch

It is designed to be small enough to train on a MacBook CPU, while still exposing the core concepts used in modern LLMs like GPT-2/GPT-3.

⸻

🔥 Features

✔ Train a custom BPE tokenizer

Using the tokenizers library, the model learns subword units that balance flexibility and efficiency.

✔ Implement a full GPT-style architecture
	•	Token embeddings
	•	Positional embeddings
	•	Multi-head self-attention
	•	Feed-forward blocks
	•	LayerNorm + residual connections
	•	Causal masking

✔ Observe training dynamics
	•	Loss curve
	•	Overfitting behavior
	•	Generated text samples
	•	Impact of context length and model capacity

✔ Fully runnable on CPU

No GPU required.