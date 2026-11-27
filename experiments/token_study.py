"""Attribute next-token loss and fit probability mixtures using validation only."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from lab.server import Lab


DATA = Path("artifacts/shakespeare-eval")
STUDY = Path("artifacts/window-study")
RESULTS = Path("results")


@torch.inference_mode()
def probabilities(lab, text):
    enc = lab.tok.encode(text, add_special_tokens=False)
    ids = enc.ids
    gpt = []
    for start in range(0, len(ids)-1, lab.model.context_length):
        stop = min(start + lab.model.context_length, len(ids)-1)
        logits, _ = lab.model(torch.tensor([ids[start:stop]]))
        targets = torch.tensor(ids[start+1:stop+1])
        gpt.extend(logits[0].softmax(-1)[torch.arange(len(targets)),targets].tolist())
    contexts = [tuple(ids[max(0,i-2):i]) for i in range(1,len(ids))]
    tri = [lab.trigram.probability(c,t) for c,t in zip(contexts,ids[1:])]
    bins = np.digitize([lab.trigram.totals[c] for c in contexts], [5,50])
    return enc, np.array(gpt), np.array(tri), bins


def attribution(text, enc, gpt, tri):
    speakers = set()
    for m in re.finditer(r"(?m)^[A-Z][A-Z '’-]+(?=:)", text):
        speakers.update(range(*m.span()))
    groups = defaultdict(list)
    for i,(a,b) in enumerate(enc.offsets[1:]):
        raw = text[a:b]
        word = raw.strip()
        if "\n" in raw:
            label = "newline"
        elif any(j in speakers for j in range(a,b)):
            label = "speaker_header"
        elif not word:
            label = "whitespace"
        elif all(not ch.isalnum() for ch in word):
            label = "punctuation"
        elif (word.isalpha() and not (a+raw.index(word)>0 and text[a+raw.index(word)-1].isalpha())
              and not (b<len(text) and text[b].isalpha())):
            label = "whole_word"
        else:
            label = "fragment_or_other"
        groups[label].append(i)
    delta = np.log(tri/gpt)
    rows = {}
    for name, ii in groups.items():
        rows[name] = dict(count=len(ii), fraction=len(ii)/len(delta),
                          mean_delta=float(delta[ii].mean()), contribution=float(delta[ii].sum()/len(delta)))
    examples = []
    for i in [int(delta.argmin()),int(delta.argmax())]:
        a,b = enc.offsets[i+1]
        examples.append(dict(prefix=text[enc.offsets[(i//32)*32][0]:a],target=text[a:b],
                             gpt=float(gpt[i]),trigram=float(tri[i])))
    assert np.isclose(sum(row["contribution"] for row in rows.values()),delta.mean())
    return dict(targets=len(delta), mean_delta=float(delta.mean()), groups=rows, examples=examples)


def loss(g,t,w):
    return float(-np.log(w*g+(1-w)*t).mean())


def main():
    lab = Lab()
    out = {"attribution": {}}
    for name,path in [("plays", DATA/"test.txt"),("sonnets", STUDY/"sonnets.txt")]:
        text = path.read_text()
        enc,g,t,_ = probabilities(lab,text)
        out["attribution"][name] = attribution(text,enc,g,t)
    _,g,t,b = probabilities(lab,(DATA/"validation.txt").read_text())
    grid = np.linspace(0,1,21)
    fixed = min(grid,key=lambda w:loss(g,t,w))
    weights = np.array([min(grid,key=lambda w:loss(g[b==i],t[b==i],w)) if (b==i).any() else fixed for i in range(3)])
    fixed_loss, bucket_loss = loss(g,t,fixed), loss(g,t,weights[b])
    chosen = "frequency" if bucket_loss < fixed_loss else "fixed"
    out["validation"] = dict(fixed_weight=fixed,bucket_weights=weights.tolist(),selected=chosen,
                             fixed_loss=fixed_loss,frequency_loss=bucket_loss)
    source = (STUDY/"venus-source.txt").read_bytes()
    text = source.decode("ascii").replace("\r\n","\n")
    body = text[text.index("EVEN as the sun"):text.index("End of the Project Gutenberg EBook")]
    lines = [re.sub(r"\s+\d+\s*$","",line).strip() for line in body.splitlines() if line.strip()]
    train_lines = {line.strip().casefold() for line in (DATA/"train.txt").read_text().splitlines()}
    _,g,t,b = probabilities(lab,"\n".join(lines)+"\n")
    out["final"] = dict(source_sha256=hashlib.sha256(source).hexdigest(),targets=len(g),lines=len(lines),
        exact_training_line_matches=sum(line.casefold() in train_lines for line in lines),
        perplexity={k:float(np.exp(loss(g,t,w))) for k,w in dict(gpt=1,trigram=0,fixed=fixed,frequency=weights[b]).items()})
    (RESULTS/"token-study.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__ == "__main__":
    main()
