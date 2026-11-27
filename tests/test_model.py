import unittest

import torch

from mini_gpt_bpe import GPTLanguageModel


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = GPTLanguageModel(24, 16, 4, 2, 0, context_length=8).eval()

    def test_future_tokens_do_not_change_past_logits(self):
        original = torch.tensor([[1, 2, 3, 4]])
        changed = torch.tensor([[1, 2, 7, 8]])
        torch.testing.assert_close(self.model(original)[0][:, :2], self.model(changed)[0][:, :2])

    def test_last_projection_matches_full_projection(self):
        for length in (1, 5, 8):
            tokens = torch.randint(0, 24, (2, length))
            full, _ = self.model(tokens)
            last, _ = self.model(tokens, last_only=True)
            torch.testing.assert_close(last, full[:, -1:])

    def test_generation_crosses_context_boundary(self):
        prompt = torch.randint(0, 24, (1, 8))
        greedy = self.model.generate(prompt, 4, temperature=0)
        top_one = self.model.generate(prompt, 4, top_k=1)
        self.assertEqual(greedy.shape, (1, 12))
        torch.testing.assert_close(greedy, top_one)
