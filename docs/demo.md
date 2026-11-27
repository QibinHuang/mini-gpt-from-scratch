# Three-minute walkthrough

**0:00–0:45 · Show a prediction.** Run the quickstart on the project homepage.
Explain how BPE maps the prompt into IDs and how the causal mask prevents a
token from reading its future. Switch to temperature zero and show the
repetition that can occur with greedy decoding.

**0:45–1:45 · Explain one model decision.** Open the window comparison on the
homepage. Each run saw 819,200 prediction targets over 1,600 optimizer
updates. Compare the spread across seeds with the gap between the means.
Explain the difference between shortening context during inference and
training with a different context. Discuss the change in batch shape.

**1:45–2:40 · Show one engineering change.** Open the generation loop in
`mini_gpt_bpe.py`. Self-attention still processes the preceding tokens. Only
the final hidden state needs the vocabulary projection to predict the next
token. Show the logit equivalence test and the ten paired timing runs.

**2:40–3:00 · Show the boundary of the result.** Compare in-domain drama and
out-of-domain poetry. Perplexity measures next-token predictions. Display a
generated passage so the audience can judge its coherence independently.
