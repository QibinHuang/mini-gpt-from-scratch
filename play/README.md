# The Missing Word · 莎士比亚缺了一词

你、一个只看前两个 token 的 trigram、一个能看前 32 个 token 的 MiniGPT，同时读到舞台上的半句话。你会押哪个词？

先在四个候选中选一个，再点开原文。选中记一分。六题来自测试文本，专挑容易辨认的 token，适合演示。模型概率取自整个词表，各行之和可以小于 100%。

想让终端计分，可以从仓库根目录运行 `python3 play/quiz.py`，无需安装依赖。

## 第 1 幕 · 戏剧

```text
ate, let's stand aside and see the end of
this controversy.

TRANIO:
Sir, ▏
```

A. **my**　·　B. **I**　·　C. **you**　·　D. **what**

<details><summary>揭晓原文与模型的选择</summary>

原文接 **what**。

```text
what are you that offer to beat my servant?
```

| 候选 | MiniGPT | Trigram |
| --- | ---: | ---: |
| my | 6.57% | 4.89% |
| I | 8.86% | 13.25% |
| you | 6.04% | 2.36% |
| what | 1.42% | 0.26% |

</details>

## 第 2 幕 · 戏剧

```text
, where is my son Lucentio?

TRANIO:
Call forth an officer.
Carry ▏
```

A. **to**　·　B. **that**　·　C. **this**　·　D. **the**

<details><summary>揭晓原文与模型的选择</summary>

原文接 **this**。

```text
this mad knave to the gaol. Father
```

| 候选 | MiniGPT | Trigram |
| --- | ---: | ---: |
| to | 0.88% | 0.28% |
| that | 1.06% | 0.25% |
| this | 0.64% | 0.16% |
| the | 2.54% | 0.09% |

</details>

## 第 3 幕 · 戏剧

```text
ATHARINA:
No, sir, God forbid; but ashamed to kiss.

PETRUCHIO:
Why, ▏
```

A. **I**　·　B. **my**　·　C. **sir**　·　D. **then**

<details><summary>揭晓原文与模型的选择</summary>

原文接 **then**。

```text
then let's home again. Come, sirrah,
```

| 候选 | MiniGPT | Trigram |
| --- | ---: | ---: |
| I | 8.92% | 3.05% |
| my | 5.74% | 1.72% |
| sir | 6.10% | 2.01% |
| then | 1.11% | 12.98% |

</details>

## 第 4 幕 · 十四行诗

```text
irest creatures we desire increase,
That thereby beauty's rose might never die,
But as the riper ▏
```

A. **to**　·　B. **with**　·　C. **should**　·　D. **than**

<details><summary>揭晓原文与模型的选择</summary>

原文接 **should**。

```text
should by time decease,
His tender heir
```

| 候选 | MiniGPT | Trigram |
| --- | ---: | ---: |
| to | 1.08% | 1.26% |
| with | 0.97% | 0.59% |
| should | 0.04% | 0.02% |
| than | 0.87% | 0.50% |

</details>

## 第 5 幕 · 十四行诗

```text
mother.
For where is she so fair whose unear'd womb
Disdains the tillage of thy husbandry?
Or ▏
```

A. **I**　·　B. **my**　·　C. **thou**　·　D. **who**

<details><summary>揭晓原文与模型的选择</summary>

原文接 **who**。

```text
who is he so fond will be the tomb,
```

| 候选 | MiniGPT | Trigram |
| --- | ---: | ---: |
| I | 1.42% | 3.49% |
| my | 1.26% | 1.33% |
| thou | 1.20% | 0.88% |
| who | 0.07% | 0.00% |

</details>

## 第 6 幕 · 十四行诗

```text
going in thy noon:
Unlook'd, on diest unless thou get a son.
Music to hear, ▏
```

A. **I**　·　B. **my**　·　C. **sir**　·　D. **why**

<details><summary>揭晓原文与模型的选择</summary>

原文接 **why**。

```text
why hear'st thou music sadly?
S
```

| 候选 | MiniGPT | Trigram |
| --- | ---: | ---: |
| I | 7.02% | 5.21% |
| my | 4.76% | 1.12% |
| sir | 5.65% | 0.81% |
| why | 0.26% | 0.06% |

</details>

## 散场之后

哪一题你是凭台词的意思选的？哪一题只靠前面两个词就够了？在全部戏剧测试位置上，MiniGPT 的困惑度是 65.97，插值 trigram 是 73.48。换成十四行诗，两者分别为 91.48 和 90.55。六道题之外的完整结果见[评估数据](../results/trigram-review.json)。
