"""Curate ten high-confidence mistakes on readable tokens, with exact prompt re-encoding."""
import json
import struct
import zlib
from pathlib import Path
import torch
from lab.server import Lab, HERE
from mini_gpt_bpe import load_tokenizer

def main():
    tok = load_tokenizer("demo/tokenizer.json")
    data = Path("artifacts/shakespeare-eval")
    def encode(text):
        return tok.encode(text, add_special_tokens=False).ids
    train = encode((data / "train.txt").read_text())
    (HERE / "training-ids.z").write_bytes(zlib.compress(struct.pack(f"<{len(train)}H", *train), 9))
    lab = Lab()
    cards = []
    for genre, path in [("戏剧", data / "test.txt"), ("十四行诗", Path("artifacts/window-study/sonnets.txt"))]:
        candidates = []
        for line in path.read_text().splitlines():
            ids = encode(line)
            for i in range(5, min(32,len(ids))):
                word = tok.decode([ids[i]])
                if not (word.startswith(" ") and word.strip().isalpha() and len(word.strip()) >= 3):
                    continue
                prompt = tok.decode(ids[:i]).lstrip()
                if encode(prompt) == ids[:i]:
                    candidates.append((prompt,ids[i],line))
                break
        ranked = []
        for prompt, target, line in candidates:
            ids = encode(prompt)
            with torch.inference_mode():
                logits,_ = lab.model(torch.tensor([ids]), last_only=True)
                probs = logits[0,-1].softmax(-1)
            guess = int(probs.argmax())
            if guess == target or not tok.decode([guess]).strip().isalpha():
                continue
            ranked.append((float(probs[guess]),prompt,target,line))
        seen=set()
        for _,prompt,target,line in sorted(ranked,reverse=True):
            if prompt in seen: continue
            original=lab.predict(prompt,target)
            choices=list(dict.fromkeys([lab.label(target)]+[v for v,_ in original["gpt"]["top"]]))[:4]
            cards.append(dict(genre=genre,prompt=prompt,target=target,answer=lab.label(target),
                              line=line,choices=sorted(choices),original=original))
            seen.add(prompt)
            if len(seen)==5: break
    assert len(cards)==10
    (HERE/"cards.json").write_text(json.dumps(cards,ensure_ascii=False,indent=2)+"\n")

if __name__ == "__main__":
    main()
