# Which predictions benefit from context?

The released MiniGPT and interpolated trigram score the same next-token targets. The model uses its native 32-token blocks. The trigram sees the two preceding IDs across block boundaries. Token-level attribution uses natural-log loss differences (GPT minus trigram). Each category contribution is its average difference multiplied by its share of targets. Contributions sum to the corpus-wide gap.

| Category | Play target share | Play contribution | Sonnet target share | Sonnet contribution |
| --- | ---: | ---: | ---: | ---: |
| Speaker headers | 10.8% | −0.2023 | 0% | 0 |
| Whole alphabetic words | 25.8% | −0.0996 | 32.8% | −0.1741 |
| Punctuation | 11.7% | −0.0272 | 9.1% | −0.0297 |
| Newlines | 10.1% | +0.0041 | 6.4% | −0.0101 |
| Other whitespace | <0.1% | −0.0003 | 0.2% | −0.0017 |
| Fragments and residual tokens | 41.5% | +0.2174 | 51.5% | +0.2257 |
| Total | 100% | **−0.1079** | 100% | **+0.0102** |

These are surface-form categories derived from tokenizer offsets. Speaker headers use uppercase lines before a colon. Contractions can span categories. In the sonnets, after “The forward vio”, trigram assigned 75.48% to “let” and GPT 0.0018%. This is an extreme loss difference. It illustrates why predicting a word fragment can favor local counts. [Counts, means and extreme examples](../results/token-study.json)

## Paired interventions

Every retained input re-encodes to exactly 32 tokens and preserves the last two IDs. The trigram therefore gives identical distributions before and after each edit. For each applicable combination below, 256 eligible positions are selected in seeded random order. The same edited prompts are evaluated on the 32-token models trained with seeds 42, 43 and 44.

| Edit | Plays, change in loss [95% interval] | Sonnets, change in loss [95% interval] |
| --- | ---: | ---: |
| Shuffle internal complete alphabetic words | +0.039 [0.010, 0.071] | +0.068 [0.031, 0.106] |
| Move a newline to a word boundary | +0.042 [0.004, 0.088] | +0.080 [0.028, 0.139] |
| Substitute one content word from a fixed dictionary | +0.0004 [−0.005, 0.005] | −0.0015 [−0.005, 0.002] |
| Replace speaker label with another training-set name | +0.0035 [−0.009, 0.017] | No speaker headers |

Positive values mean the original continuation became harder to predict. Intervals resample occupied contiguous 256-token blocks 2,000 times, after averaging paired differences across seeds. They describe uncertainty over sampled text, conditional on these checkpoints. Eligible positions differ between intervention types. The speaker experiment changes a previous label while predicting a later token. The attribution table scores tokens inside the label itself. Word shuffles may disrupt syntax and BPE boundaries. [Per-seed results](../results/context-edits.json)

## Can complementary errors improve prediction?

Mix the two normalized probability distributions before taking the logarithm. Fit GPT weights only on the play validation split, using a grid from 0 to 1 in steps of 0.05. Compare a fixed weight against three weights grouped by training frequency of the preceding two-token context (0–4, 5–49, 50+ occurrences). The fixed weight was 0.40. The three weights were 0.50, 0.40 and 0.45. Validation selected the bucketed rule by a small margin (3.5567 versus 3.5582).

The prespecified external work was Shakespeare’s *Venus and Adonis*, Gutenberg ebook 1045. Remove front matter, line numbers, indentation and blank lines, retaining 1,194 verse lines and 19,789 prediction targets. Zero full lines matched the training corpus after trimming and case folding. This is a single additional work by the same author.

| Final predictor | External perplexity |
| --- | ---: |
| MiniGPT | 100.36 |
| Interpolated trigram | 95.79 |
| Fixed mixture | 70.93 |
| Validation-selected frequency mixture | **70.56** |

Most of the gain came from mixing. Adding the frequency buckets improved perplexity by about 0.5% relative to the fixed mixture. The released GPT checkpoint and historical results are still individual-model measurements. [External results and source digest](../results/token-study.json)

## Reproduce

Prepare the corpora and the six window-study checkpoints using [the existing instructions](reproduce.md). From the repository root:

```sh
curl -fL https://raw.githubusercontent.com/GITenberg/Venus-and-Adonis_1045/master/1045.txt -o artifacts/window-study/venus-source.txt
OMP_NUM_THREADS=2 python -m experiments.token_study
OMP_NUM_THREADS=2 python -m experiments.context_edits
```
