"""Play the next-token challenge without installing dependencies."""
import json
from pathlib import Path


def main():
    rounds = json.loads(Path(__file__).with_name("rounds.json").read_text())
    score = 0
    for number, row in enumerate(rounds, 1):
        print(f"\n{number}/{len(rounds)} · {row['genre']}\n{row['prompt']} ▏")
        for i, word in enumerate(row["options"], 1):
            print(f"  {i}. {word}")
        choice = input("你的选择 [1–4，q 退出] ").strip()
        while choice not in ("1", "2", "3", "4", "q"):
            choice = input("请输入 1–4 或 q：").strip()
        if choice == "q":
            return
        answer = row["answer"]
        score += int(int(choice) == answer + 1)
        print(f"原文续写：{row['continuation']}\n正确 token：{row['options'][answer]}")
        print(f"MiniGPT 给原文的概率 {row['gpt'][answer]:.1%} · Trigram {row['trigram'][answer]:.1%}")
    print(f"得分 {score}/{len(rounds)}")


if __name__ == "__main__":
    main()
