"""Serve an actually trained G1 policy to a CPU simulator over loopback HTTP."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from robot_foundry.g1_policy import G1Predictor
from robot_foundry.g1_transport import decode_request
from robot_foundry.train import seed_everything


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backend", choices=["smolvla", "tiny_bc"], default="smolvla")
    p.add_argument("--device", default="auto")
    p.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    p.add_argument("--port", type=int, default=8775)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(2)
    policy = G1Predictor(args.checkpoint, args.backend, args.device, args.precision)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, policy.metadata()) if self.path == "/health" else self.reply(
                404, {"error": "Unknown endpoint"}
            )

        def do_POST(self):
            if self.path != "/predict":
                return self.reply(404, {"error": "Unknown endpoint"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size < 2 * 2**20:
                    raise ValueError("Invalid request size")
                state, image, prompt = decode_request(json.loads(self.rfile.read(size)), policy.image_size)
                self.reply(200, {"tokens": policy.predict(state, image, prompt).tolist()})
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, {"error": str(error)})
            except FloatingPointError as error:
                self.reply(500, {"error": str(error)})

        def log_message(self, *_):
            pass

    print(f"Trained G1 policy ready on 127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
