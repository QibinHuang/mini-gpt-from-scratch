"""Compare greedy generation with all-position and last-position vocabulary projection."""
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from mini_gpt_bpe import GPTLanguageModel

torch.set_num_threads(4)
torch.manual_seed(42)
ckpt = torch.load("demo/model.pt", map_location="cpu", weights_only=True)
model = GPTLanguageModel(**ckpt["config"]).eval()
model.load_state_dict(ckpt["model_state_dict"])
prompt = torch.randint(0, ckpt["config"]["vocab_size"], (1, model.context_length))
runs = {"full": [], "last": []}
with torch.inference_mode():
    full, _ = model(prompt)
    last, _ = model(prompt, last_only=True)
    torch.testing.assert_close(last, full[:, -1:], atol=1e-5, rtol=1e-5)
    for iteration in range(12):
        outputs = []
        order = ("full", "last") if iteration % 2 == 0 else ("last", "full")
        for mode in order:
            sequence = prompt
            start = perf_counter()
            for _ in range(128):
                logits, _ = model(sequence[:, -model.context_length:], last_only=mode == "last")
                sequence = torch.cat((sequence, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
            elapsed = perf_counter() - start
            if iteration >= 2:
                runs[mode].append(128 / elapsed)
            outputs.append(sequence)
        torch.testing.assert_close(*outputs)
rates = {name: median(values) for name, values in runs.items()}
Path("results/benchmark.json").write_text(json.dumps(dict(context=model.context_length,
    torch_version=str(torch.__version__), threads=4, tokens_per_second=runs, median=rates,
    speedup=rates["last"]/rates["full"], max_abs_error=(last-full[:, -1:]).abs().max().item()), indent=2)+"\n")
