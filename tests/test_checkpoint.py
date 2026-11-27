from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from unittest.mock import patch

from tokenizers import Tokenizer

from generate_bpe import main


ROOT = Path(__file__).resolve().parents[1]


def generate(checkpoint, tokenizer):
    args = ["generate_bpe.py", "--checkpoint", str(checkpoint), "--tokenizer", str(tokenizer),
            "--prompt", "Language models", "--tokens", "2", "--temperature", "0"]
    output = StringIO()
    with patch.object(sys, "argv", args), redirect_stdout(output):
        main()
    return output.getvalue()



class CheckpointTests(unittest.TestCase):
    def test_release_rejects_same_size_tokenizer_with_swapped_ids(self):
        original = ROOT / "demo" / "tokenizer.json"
        data = json.loads(original.read_text())
        vocab = data["model"]["vocab"]
        first, second = sorted(vocab, key=vocab.get)[-2:]
        vocab[first], vocab[second] = vocab[second], vocab[first]
        with TemporaryDirectory() as directory:
            wrong = Path(directory) / "tokenizer.json"
            wrong.write_text(json.dumps(data))
            self.assertEqual(Tokenizer.from_file(str(wrong)).get_vocab_size(), len(vocab))
            with self.assertRaisesRegex(ValueError, "does not match the checkpoint fingerprint"):
                generate(ROOT / "demo" / "model.pt", wrong)

    def test_original_checkpoint_still_generates(self):
        result = generate(ROOT / "mini_gpt_bpe.pt", ROOT / "tokenizer.json")
        self.assertTrue(result.strip().startswith("Language models"))
