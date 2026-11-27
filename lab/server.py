"""Run from the repo root with python -m lab.server."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import struct
import zlib

import torch

from experiments.compare_trigram import Trigram
from mini_gpt_bpe import GPTLanguageModel, load_tokenizer


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "lab"


class Lab:
    def __init__(self):
        ckpt = torch.load(ROOT / "demo/model.pt", map_location="cpu", weights_only=True)
        self.model = GPTLanguageModel(**ckpt["config"]).eval()
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.tok = load_tokenizer(ROOT / "demo/tokenizer.json", ckpt["tokenizer_sha256"])
        raw = zlib.decompress((HERE / "training-ids.z").read_bytes())
        self.trigram = Trigram([v for v, in struct.iter_unpack("<H", raw)], self.tok.get_vocab_size())

    def label(self, i):
        return self.tok.decode([i], skip_special_tokens=False).replace("\n", "↵").replace(" ", "·")

    @torch.inference_mode()
    def predict(self, text, target):
        ids = self.tok.encode(text, add_special_tokens=False).ids[-self.model.context_length:]
        logits, _ = self.model(torch.tensor([ids]), last_only=True)
        probs = logits[0,-1].softmax(-1).tolist()
        suffix = tuple(ids[-2:])
        tri = [self.trigram.probability(suffix, j) for j in range(len(probs))]
        def score(values):
            top = sorted(range(len(values)), key=values.__getitem__, reverse=True)[:4]
            return dict(target=values[target], top=[[self.label(i), values[i]] for i in top])
        return dict(gpt=score(probs), tri=score(tri), suffix=list(suffix),
                    labels=[self.label(i) for i in suffix])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    torch.set_num_threads(2)
    lab = Lab()
    cards = json.loads((HERE / "cards.json").read_text())

    class Handler(BaseHTTPRequestHandler):
        def reply(self, data, mime, status=200):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/cards":
                return self.reply(json.dumps(cards).encode(), "application/json")
            if self.path == "/":
                return self.reply((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            self.send_error(404)

        def do_POST(self):
            if self.path != "/predict":
                return self.send_error(404)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 24000:
                    raise ValueError
                payload = json.loads(self.rfile.read(size))
                index, text = payload["card"], payload["text"]
                if type(index) is not int or not 0 <= index < len(cards):
                    raise ValueError
                if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                    raise ValueError
            except (ValueError, TypeError, KeyError):
                return self.send_error(400)
            self.reply(json.dumps(lab.predict(text, cards[index]["target"])).encode(), "application/json")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"http://127.0.0.1:{server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
