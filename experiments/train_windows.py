"""Run the six equal-token-budget Tiny Shakespeare experiments from the repo root."""
import os
from pathlib import Path
import subprocess
import sys
import time

data = Path("artifacts/shakespeare-eval")
root = Path("artifacts/window-study")
for seed in (42, 43, 44):
    for length in (32, 64):
        output = root / f"ctx{length}-seed{seed}"
        output.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with (output / "training.log").open("w") as log:
            subprocess.run([
                sys.executable, "-u", "mini_gpt_bpe.py", "--input", str(data / "train.txt"),
                "--validation", str(data / "validation.txt"), "--tokenizer", str(data / "tokenizer.json"),
                "--context-length", str(length), "--batch-size", str(512 // length),
                "--steps", "1600", "--eval-interval", "200", "--seed", str(seed),
                "--output", str(output / "final.pt"), "--best-output", str(output / "best.pt"),
            ], stdout=log, check=True, env={**os.environ, "OMP_NUM_THREADS": "4"})
        (output / "elapsed.txt").write_text(str(time.perf_counter() - start))
        print(f"Completed context {length}, seed {seed}", flush=True)
