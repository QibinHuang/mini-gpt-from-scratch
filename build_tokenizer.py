"""
build_tokenizer.py
从 input.txt 训练一个 BPE tokenizer，并保存为 tokenizer.json
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

def main():
    input_file = "input.txt"

    # 1. 构建一个空的 BPE 模型
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 使用 ByteLevel 预分词（对英文很常用）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # 2. 定义 BPE 训练器
    vocab_size = 2000  # 对你现在的小文本够用了，后续可以调大
    special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # 3. 调用 train 在 input.txt 上训练
    tokenizer.train(files=[input_file], trainer=trainer)

    # 4. 设置 BOS/EOS 模板（可选，但对语言模型有用）
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[
            ("[BOS]", bos_id),
            ("[EOS]", eos_id),
        ],
    )

    # 5. 保存到文件
    tokenizer.save("tokenizer.json")
    print("Tokenizer saved to tokenizer.json")
    print("Vocab size:", tokenizer.get_vocab_size())

if __name__ == "__main__":
    main()