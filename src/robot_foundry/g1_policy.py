"""Load a trained G1 latent policy; never fall back to an untrained Panda/base policy."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from .backends import SmolBackend
from .data import tokenize
from .g1_schema import live_state, load_g1_statistics, validate_g1_manifest
from .io import file_hash, read_json
from .model import TinyChunkPolicy
from .precision import autocast
from .train import to_device


class G1Predictor:
    def __init__(self, checkpoint, backend="smolvla", device="auto", precision="auto"):
        self.root = Path(checkpoint).resolve()
        self.manifest = read_json(self.root / "data_manifest.json")
        validate_g1_manifest(self.manifest)
        self.training_report = read_json(self.root.parent / "report.json")
        if (
            self.training_report.get("optimizer_updates", 0) < 1
            or self.training_report["action_dimension"] != 64
        ):
            raise ValueError("Inference requires a G1 checkpoint produced by successful optimizer updates")
        if self.training_report["backend"] != backend:
            raise ValueError("Backend differs from the trained checkpoint")
        if self.training_report["data_fingerprint"] != self.manifest["fingerprint"]:
            raise ValueError("Training report and checkpoint dataset identity differ")
        self.device = to_device({"device": device})
        self.precision = (
            ("fp16" if backend == "smolvla" and self.device.type == "cuda" else "fp32")
            if precision == "auto"
            else precision
        )
        self.backend_name = backend
        self.horizon = self.manifest["config"]["horizon"]
        self.image_size = self.manifest["config"]["image_size"]
        self.stats = load_g1_statistics(self.root, self.manifest)
        self.state_mean = np.asarray(self.stats["observation.state"]["mean"], np.float32)
        self.state_std = np.asarray(self.stats["observation.state"]["std"], np.float32)
        self.action_mean = np.asarray(self.stats["action"]["mean"], np.float32)
        self.action_std = np.asarray(self.stats["action"]["std"], np.float32)
        if backend == "tiny_bc":
            saved = torch.load(self.root / "model.pt", map_location="cpu", weights_only=True)
            if saved["spec"]["state_dim"] != 70 or saved["spec"]["action_dim"] != 64:
                raise ValueError("TinyBC model dimensions do not match G1")
            self.model = TinyChunkPolicy(**saved["spec"]).to(self.device)
            self.model.load_state_dict(saved["model"])
            self.model.requires_grad_(False).eval()
            weight_path = self.root / "model.pt"
        else:
            cfg = {
                "inference_only": True,
                "precision": self.precision,
                "output_dir": str(self.root.parent),
                "registry": "configs/datasets.json",
            }
            self.backend = SmolBackend(
                cfg, SimpleNamespace(root=self.root, manifest=self.manifest), self.device, load_path=self.root
            )
            self.model = self.backend.model
            weight_path = self.root / "model.safetensors"
        self.weights_sha256 = file_hash(weight_path)

    def metadata(self):
        return {
            "ready": True,
            "backend": self.backend_name,
            "robot": self.manifest["signature"]["robot_type"],
            "action_dimension": 64,
            "state_dimension": 70,
            "horizon": self.horizon,
            "image_size": self.image_size,
            "precision": self.precision,
            "device": str(self.device),
            "data_fingerprint": self.manifest["fingerprint"],
            "controller_identity": self.manifest["controller_identity"],
            "optimizer_updates": self.training_report["optimizer_updates"],
            "weights_sha256": self.weights_sha256,
        }

    @torch.inference_mode()
    def predict(self, state, image, prompt):
        state, image = live_state(state), np.asarray(image, np.uint8)
        if image.shape != (self.image_size, self.image_size, 3):
            raise ValueError("G1 camera resolution differs from training")
        raw = {
            "images": torch.from_numpy(image.transpose(2, 0, 1).copy())[None, None],
            "state": torch.from_numpy(
                (state - self.state_mean) / self.state_std if self.backend_name == "tiny_bc" else state
            )[None],
            "prompt": [prompt],
            "tokens": torch.from_numpy(tokenize(prompt))[None],
            "action": torch.zeros(1, self.horizon, 64),
            "valid": torch.ones(1, self.horizon, dtype=torch.bool),
        }
        with autocast(self.device, self.precision):
            if self.backend_name == "tiny_bc":
                pred = self.model(
                    {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
                )
            else:
                pred = self.backend.predict(self.backend.process(raw))
        tokens = pred[0].float().cpu().numpy() * self.action_std + self.action_mean
        if tokens.shape != (self.horizon, 64) or not np.isfinite(tokens).all():
            raise FloatingPointError("Invalid G1 latent action chunk")
        return tokens
