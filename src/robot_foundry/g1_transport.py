"""Loopback-only G1 image/state RPC independent of PyTorch in the simulator process."""

import base64
import io
import json
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image
from .g1_schema import live_state


def encode_request(state, image, prompt):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return {
        "state": np.asarray(state).tolist(),
        "image": base64.b64encode(buffer.getvalue()).decode(),
        "prompt": prompt,
    }


def decode_request(payload, image_size):
    state = live_state(payload["state"])
    prompt = payload["prompt"]
    if (
        state.shape != (70,)
        or not np.isfinite(state).all()
        or not isinstance(prompt, str)
        or not 0 < len(prompt) <= 1000
    ):
        raise ValueError("Invalid G1 state/prompt")
    with Image.open(io.BytesIO(base64.b64decode(payload["image"], validate=True))) as picture:
        if picture.size != (image_size, image_size):
            raise ValueError("G1 image size differs from checkpoint")
        image = np.asarray(picture.convert("RGB")).copy()
    return state, image, prompt


class RemoteG1Policy:
    def __init__(self, url, manifest):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ["127.0.0.1", "localhost"]:
            raise ValueError("G1 simulation connects only to loopback HTTP")
        self.url = url.rstrip("/")
        with urllib.request.urlopen(self.url + "/health", timeout=30) as response:
            self.health = json.load(response)
        expected = {
            "action_dimension": 64,
            "state_dimension": 70,
            "horizon": manifest["config"]["horizon"],
            "image_size": manifest["config"]["image_size"],
            "data_fingerprint": manifest["fingerprint"],
            "controller_identity": manifest["controller_identity"],
        }
        if (
            any(self.health.get(k) != v for k, v in expected.items())
            or self.health.get("optimizer_updates", 0) < 1
        ):
            raise ValueError("Policy server has not loaded the matching trained G1 policy/controller")
        self.horizon = self.health["horizon"]

    def predict(self, state, image, prompt):
        request = urllib.request.Request(
            self.url + "/predict",
            data=json.dumps(encode_request(state, image, prompt)).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            tokens = np.asarray(json.load(response)["tokens"], np.float32)
        if tokens.shape != (self.horizon, 64) or not np.isfinite(tokens).all():
            raise ValueError("G1 policy returned an invalid latent chunk")
        return tokens
