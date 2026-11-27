"""Select a window on validation, using the full split and each model's native window."""
import hashlib
import json
from pathlib import Path
from shutil import copyfile
from statistics import mean, stdev

import torch

from evaluate_bpe import model_loss
from mini_gpt_bpe import GPTLanguageModel, load_tokenizer

data = Path("artifacts/shakespeare-eval")
root = Path("artifacts/window-study")
results = Path("results")
results.mkdir(exist_ok=True)
tokenizer = load_tokenizer(data / "tokenizer.json")
tokens = torch.tensor(tokenizer.encode((data / "validation.txt").read_text(), add_special_tokens=False).ids)
groups = []
for length in (32, 64):
    runs = []
    for seed in (42, 43, 44):
        folder = root / f"ctx{length}-seed{seed}"
        ckpt = torch.load(folder / "best.pt", weights_only=True, map_location="cpu")
        model = GPTLanguageModel(**ckpt["config"]).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        elapsed = float((folder / "elapsed.txt").read_text())
        loss = model_loss(model, tokens)
        runs.append(dict(seed=seed, step=ckpt["model_step"], validation_loss=loss,
                         seconds=elapsed, tokens_per_second=819200 / elapsed))
    losses = [run["validation_loss"] for run in runs]
    groups.append(dict(context=length, runs=runs, mean_loss=mean(losses), sd_loss=stdev(losses)))
winner = min(groups, key=lambda group: group["mean_loss"])["context"]
release = Path("demo")
release.mkdir(exist_ok=True)
chosen = torch.load(root / f"ctx{winner}-seed42/best.pt", weights_only=True, map_location="cpu")
chosen["tokenizer_sha256"] = hashlib.sha256((data / "tokenizer.json").read_bytes()).hexdigest()
torch.save(chosen, release / "model.pt")
copyfile(data / "tokenizer.json", release / "tokenizer.json")
(results / "windows.json").write_text(json.dumps(dict(groups=groups, selected_context=winner,
    release_seed=42, validation_targets=len(tokens)-1), indent=2) + "\n")
