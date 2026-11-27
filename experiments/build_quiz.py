"""Build six illustrative next-token rounds from held-out Shakespeare excerpts."""
import json
from pathlib import Path
import random

import torch

from experiments.compare_trigram import Trigram
from mini_gpt_bpe import GPTLanguageModel, load_tokenizer


def main():
    ckpt = torch.load("demo/model.pt", weights_only=True, map_location="cpu")
    model = GPTLanguageModel(**ckpt["config"]).eval()
    model.load_state_dict(ckpt["model_state_dict"])
    tok = load_tokenizer("demo/tokenizer.json", ckpt["tokenizer_sha256"])
    data = Path("artifacts/shakespeare-eval")
    def encode(text):
        return tok.encode(text, add_special_tokens=False).ids

    size = tok.get_vocab_size()
    tri = Trigram(encode((data / "train.txt").read_text()), size)
    rng = random.Random(42)
    rounds = []
    with torch.inference_mode():
        for genre, path in [("戏剧", data / "test.txt"), ("十四行诗", Path("artifacts/window-study/sonnets.txt"))]:
            ids = encode(path.read_text())
            for start in (0, 500, 1500):
                for i in range(start + 32, start + 250):
                    prefix = ids[i-32:i]
                    target = ids[i]
                    answer = tok.decode([target])
                    if not (answer.startswith(" ") and answer.strip().isalpha() and len(answer.strip()) >= 3):
                        continue
                    logits, _ = model(torch.tensor([prefix]), last_only=True)
                    probs = logits[0, -1].softmax(-1)
                    choices = [target]
                    for idx in probs.argsort(descending=True).tolist():
                        label = tok.decode([idx])
                        if idx != target and label.startswith(" ") and label.strip().isalpha():
                            choices.append(idx)
                        if len(choices) == 4:
                            break
                    rng.shuffle(choices)
                    rounds.append(dict(genre=genre, prompt=tok.decode(prefix).lstrip(), answer=choices.index(target),
                        options=[tok.decode([j]).strip() for j in choices],
                        gpt=[float(probs[j]) for j in choices],
                        trigram=[tri.probability(tuple(prefix[-2:]), j) for j in choices],
                        continuation=tok.decode(ids[i:i+12])))
                    break
    assert len(rounds) == 6
    Path("play/rounds.json").write_text(json.dumps(rounds, ensure_ascii=False, indent=2)+"\n")

    page = ["# The Missing Word · 莎士比亚缺了一词", "",
            "你、一个只看前两个 token 的 trigram、一个能看前 32 个 token 的 MiniGPT，同时读到舞台上的半句话。你会押哪个词？",
            "", "先在四个候选中选一个，再点开原文。选中记一分。六题来自测试文本，专挑容易辨认的 token，适合演示。模型概率取自整个词表，各行之和可以小于 100%。",
            "", "想让终端计分，可以从仓库根目录运行 `python3 play/quiz.py`，无需安装依赖。"]
    for number, row in enumerate(rounds, 1):
        page.extend(["", f"## 第 {number} 幕 · {row['genre']}", "", "```text",
                     row["prompt"] + " ▏", "```", "",
                     "　·　".join(f"{letter}. **{word}**" for letter, word in zip("ABCD", row["options"])),
                     "", "<details><summary>揭晓原文与模型的选择</summary>", "",
                     f"原文接 **{row['options'][row['answer']]}**。", "", "```text",
                     row["continuation"].strip(), "```", "", "| 候选 | MiniGPT | Trigram |",
                     "| --- | ---: | ---: |"])
        for word, gpt, trigram in zip(row["options"], row["gpt"], row["trigram"]):
            page.append(f"| {word} | {gpt:.2%} | {trigram:.2%} |")
        page.extend(["", "</details>"])
    page.extend(["", "## 散场之后", "", "哪一题你是凭台词的意思选的？哪一题只靠前面两个词就够了？在全部戏剧测试位置上，MiniGPT 的困惑度是 65.97，插值 trigram 是 73.48。换成十四行诗，两者分别为 91.48 和 90.55。六道题之外的完整结果见[评估数据](../results/trigram-review.json)。"])
    Path("play/README.md").write_text("\n".join(page) + "\n")


if __name__ == "__main__":
    main()
