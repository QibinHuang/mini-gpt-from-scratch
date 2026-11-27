"""Paired input perturbations at fixed length and suffix, scored across training seeds."""
from collections import defaultdict
import json
import re
import random

import numpy as np
import torch
import torch.nn.functional as F

from experiments.token_study import DATA, STUDY, RESULTS
from mini_gpt_bpe import GPTLanguageModel, load_tokenizer


WORDS = re.compile(r"\b[a-zA-Z]+\b")
SUBSTITUTIONS = dict(king="lord", lord="king", love="hate", heart="mind", life="death", man="boy",
                     eyes="ears", hand="face", come="look", good="poor", sweet="fair", night="light",
                     sun="day", world="earth", time="wind", fear="hope", name="word", speak="think")


def edits(text, rng, speakers):
    # Keep both boundary words untouched in whole-word edits.
    words = list(WORDS.finditer(text))[1:-1]
    if len(words) >= 3:
        replacements = [m.group() for m in words]
        rng.shuffle(replacements)
        mixed = text
        for m,new in reversed(list(zip(words,replacements))):
            mixed = mixed[:m.start()]+new+mixed[m.end():]
        yield "whole_word_shuffle", mixed
    for m in words:
        if m.group() in SUBSTITUTIONS:
            yield "content_word", text[:m.start()]+SUBSTITUTIONS[m.group()]+text[m.end():]
            break
    match = re.search(r"(?<=\n)([A-Z][A-Z ]+)(?=:)",text)
    if match:
        for name in rng.sample(speakers,min(10,len(speakers))):
            yield "speaker",text[:match.start()]+name+text[match.end():]
    newline = text.find("\n")
    spaces = [m.start() for m in re.finditer(r"(?<=[A-Za-z]) (?=[A-Za-z])",text)]
    if newline>=0 and spaces:
        chars = list(text)
        space = rng.choice(spaces)
        chars[newline],chars[space] = chars[space],chars[newline]
        yield "move_newline","".join(chars)


def pairs(tok,text,speakers):
    ids = tok.encode(text,add_special_tokens=False).ids
    rng = random.Random(731)
    groups = defaultdict(list)
    for i in rng.sample(range(32,len(ids)),len(ids)-32):
        original = ids[i-32:i]
        prompt = tok.decode(original)
        if tok.encode(prompt,add_special_tokens=False).ids != original:
            continue
        seen = set()
        for key,edit in edits(prompt,rng,speakers):
            if key in seen or len(groups[key])>=256:
                continue
            changed = tok.encode(edit,add_special_tokens=False).ids
            if len(changed)==32 and original[-2:]==changed[-2:] and original!=changed:
                groups[key].append((i,original,changed,ids[i]))
                seen.add(key)
        expected = ("whole_word_shuffle","content_word","move_newline") + (("speaker",) if re.search(r"(?m)^[A-Z ]+:$",text[:3000]) else ())
        if all(len(groups[key])>=256 for key in expected):
            break
    return groups


@torch.inference_mode()
def differences(model,rows):
    values = []
    for start in range(0,len(rows),32):
        batch=rows[start:start+32]
        losses=[]
        for column in (1,2):
            logits,_=model(torch.tensor([r[column] for r in batch]),last_only=True)
            losses.append(F.cross_entropy(logits[:,-1],torch.tensor([r[3] for r in batch]),reduction="none"))
        values.extend((losses[1]-losses[0]).tolist())
    return values


def main():
    tok=load_tokenizer("demo/tokenizer.json")
    speakers=sorted(set(re.findall(r"(?m)^([A-Z][A-Z ]+):$",(DATA/"train.txt").read_text())))
    results={}
    for name,path in [("plays",DATA/"test.txt"),("sonnets",STUDY/"sonnets.txt")]:
        groups=pairs(tok,path.read_text(),speakers)
        results[name]={}
        for key,rows in groups.items():
            if not rows:
                continue
            per_seed=[]
            for seed in (42,43,44):
                ck=torch.load(STUDY/f"ctx32-seed{seed}/best.pt",weights_only=True,map_location="cpu")
                m=GPTLanguageModel(**ck["config"]).eval()
                m.load_state_dict(ck["model_state_dict"])
                per_seed.append(differences(m,rows))
            delta=np.mean(per_seed,axis=0)
            blocks=np.array([r[0]//256 for r in rows])
            counts=np.unique(blocks)
            totals=np.array([delta[blocks==block].sum() for block in counts])
            sizes=np.array([(blocks==block).sum() for block in counts])
            resample=np.random.default_rng(42).integers(len(counts),size=(2000,len(counts)))
            boot=totals[resample].sum(1)/sizes[resample].sum(1)
            results[name][key]=dict(targets=len(rows),blocks=len(counts),mean_delta=delta.mean(),
                seed_deltas=np.mean(per_seed,axis=1).tolist(),block_interval=np.quantile(boot,[.025,.975]).tolist())
    (RESULTS/"context-edits.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
