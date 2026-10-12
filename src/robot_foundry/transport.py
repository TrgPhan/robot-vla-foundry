"""Loopback-only simulator transport for separate NumPy training/simulation environments."""

import base64
import io
import json
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

IMAGE_KEYS = ["agentview_image", "robot0_eye_in_hand_image"]
STATE_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def encode_observation(obs, prompt):
    payload = {"prompt": prompt, "state": {k: np.asarray(obs[k]).tolist() for k in STATE_KEYS}, "images": {}}
    for key in IMAGE_KEYS:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(obs[key], np.uint8)).save(buf, format="PNG")
        payload["images"][key] = base64.b64encode(buf.getvalue()).decode("ascii")
    return payload


def decode_observation(payload):
    expected = {"robot0_eef_pos": (3,), "robot0_eef_quat": (4,), "robot0_gripper_qpos": (2,)}
    obs = {k: np.asarray(payload["state"][k], np.float32) for k in STATE_KEYS}
    if any(obs[k].shape != shape or not np.isfinite(obs[k]).all() for k, shape in expected.items()):
        raise ValueError("Invalid simulator state.")
    if np.linalg.norm(obs["robot0_eef_quat"]) < 1e-8:
        raise ValueError("Zero quaternion.")
    for key in IMAGE_KEYS:
        binary = base64.b64decode(payload["images"][key], validate=True)
        with Image.open(io.BytesIO(binary)) as img:
            if img.size != (256, 256):
                raise ValueError("Expected 256x256 simulation camera image.")
            obs[key] = np.asarray(img.convert("RGB")).copy()
    prompt = payload["prompt"]
    if not isinstance(prompt, str) or not prompt or len(prompt) > 1000:
        raise ValueError("Invalid prompt.")
    return obs, prompt


class RemotePredictor:
    def __init__(self, url, manifest):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ["127.0.0.1", "localhost"]:
            raise ValueError("The simulator transport only connects to a loopback HTTP endpoint.")
        self.url = url.rstrip("/")
        self.horizon = manifest["config"]["horizon"]
        with urllib.request.urlopen(self.url + "/health", timeout=30) as response:
            status = json.load(response)
        if status["horizon"] != self.horizon or status["data_fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Policy server loaded a different dataset/action layout.")

    def predict(self, obs, prompt):
        req = urllib.request.Request(
            self.url + "/predict",
            data=json.dumps(encode_observation(obs, prompt)).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            actions = np.asarray(json.load(response)["actions"], np.float32)
        if actions.shape != (self.horizon, 7) or not np.isfinite(actions).all():
            raise ValueError("Invalid model action chunk returned by the policy server.")
        return actions
