"""Serve a local simulation policy; binds 127.0.0.1 and sends no robot commands."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from robot_foundry.simulation import CheckpointPredictor
from robot_foundry.transport import decode_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=["tiny_bc", "smolvla"], default="smolvla")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    predictor = CheckpointPredictor(args.checkpoint, args.backend, args.device, args.precision)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                return self.reply(404, {"error": "Unknown endpoint"})
            self.reply(
                200,
                {
                    "ready": True,
                    "backend": args.backend,
                    "precision": predictor.precision,
                    "horizon": predictor.horizon,
                    "data_fingerprint": predictor.manifest["fingerprint"],
                },
            )

        def do_POST(self):
            if self.path != "/predict":
                return self.reply(404, {"error": "Unknown endpoint"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 2 * 1024 * 1024:
                    raise ValueError("Invalid request size.")
                obs, prompt = decode_observation(json.loads(self.rfile.read(length)))
                actions = predictor.predict(obs, prompt)
                self.reply(200, {"actions": actions.tolist()})
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {"error": str(error)})

        def log_message(self, *_):
            pass

    print(f"Local policy server ready: http://127.0.0.1:{args.port}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
