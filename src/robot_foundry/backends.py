"""Backend adapters; the SmolVLA backend uses LeRobot's actual pretrained policy."""

import shutil
from pathlib import Path

import torch

from .hardware import require_gpu
from .io import read_json, write_json
from .model import TinyChunkPolicy


class TinyBackend:
    def __init__(self, cfg, dataset, device):
        s = dataset.manifest["signature"]
        self.model = TinyChunkPolicy(
            s["state"]["shape"][0],
            s["action"]["shape"][0],
            len(s["cameras"]),
            dataset.manifest["config"]["horizon"],
        ).to(device)
        self.device = device
        self.velocity_weight = cfg.get("velocity_weight", 0.05)

    def process(self, batch):
        return {
            k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def loss(self, batch):
        from .data import masked_loss

        return masked_loss(self.model(batch), batch["action"], batch["valid"], self.velocity_weight)

    def predict(self, batch):
        return self.model(batch)

    def export(self, path, dataset, asset_destination=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "spec": self.model.spec}, path / "model.pt")
        write_json(path / "stats.json", read_json(dataset.root / "stats.json"))
        write_json(path / "data_manifest.json", dataset.manifest)


class SmolBackend:
    def __init__(self, cfg, dataset, device, load_path=None):
        if device.type == "cpu" and cfg.get("cpu_smoke_only") and cfg.get("steps", 0) <= 2:
            pass  # Explicit one/two-step API validation, not a CPU training recommendation.
        elif device.type == "cuda":
            require_gpu(
                "smolvla",
                device_ids=[device.index if device.index is not None else torch.cuda.current_device()],
            )
        else:
            raise RuntimeError("Use a GPU for the SmolVLA training profile.")
        from huggingface_hub import snapshot_download
        from lerobot.configs import FeatureType, PolicyFeature
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        registry = read_json(cfg.get("registry", "configs/datasets.json"))
        base = registry[cfg.get("smolvla_revision_key", "smolvla")]
        vlm = registry["smolvlm"]
        self.assets = Path(cfg["output_dir"]).resolve() / "vlm_assets"
        if load_path:
            pretrained = str(Path(load_path).resolve())
            self.assets = Path(load_path).resolve() / "vlm_assets"
            policy_cfg = SmolVLAConfig.from_pretrained(pretrained)
        else:
            pretrained = base["repo_id"]
            policy_cfg = SmolVLAConfig.from_pretrained(pretrained, revision=base["revision"])
            asset_marker = self.assets / "foundry_revision.json"
            if not asset_marker.exists() or read_json(asset_marker) != vlm:
                snapshot_download(
                    vlm["repo_id"],
                    revision=vlm["revision"],
                    ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.onnx"],
                    local_dir=str(self.assets),
                )
                write_json(asset_marker, vlm)
        s = dataset.manifest["signature"]
        size = dataset.manifest["config"]["image_size"]
        horizon = dataset.manifest["config"]["horizon"]
        policy_cfg.input_features = {
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=tuple(s["state"]["shape"]))
        }
        policy_cfg.input_features.update(
            {key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, size, size)) for key in s["cameras"]}
        )
        policy_cfg.output_features = {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=tuple(s["action"]["shape"]))
        }
        policy_cfg.device = str(device)
        policy_cfg.chunk_size = horizon
        policy_cfg.n_action_steps = min(8, horizon)
        policy_cfg.vlm_model_name = str(self.assets)
        policy_cfg.load_vlm_weights = False  # full pretrained SmolVLA weights load below
        policy_cfg.freeze_vision_encoder = True
        policy_cfg.train_expert_only = True
        policy_cfg.train_state_proj = True
        self.model = SmolVLAPolicy.from_pretrained(
            pretrained, config=policy_cfg, revision=None if load_path else base["revision"]
        )
        if cfg.get("precision") in ["fp16", "fp32"]:
            # FP32 master weights: the upstream VLM defaults to BF16, unsupported by T4 Tensor Cores.
            # AMP casts eligible operations to FP16; trainable gradients stay FP32 for GradScaler.
            self.model.float()
        self.model.to(device)
        stats = {
            key: {k: torch.tensor(v) for k, v in values.items()}
            for key, values in read_json(dataset.root / "stats.json").items()
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(policy_cfg, dataset_stats=stats)
        self.cameras = s["cameras"]
        self.device = device

    def process(self, batch):
        images = batch["images"].to(self.device, non_blocking=True).float() / 255
        raw = {
            "observation.state": batch["state"].to(self.device, non_blocking=True),
            "action": batch["action"].to(self.device, non_blocking=True),
            "task": batch["prompt"],
            "action_is_pad": ~batch["valid"].to(self.device, non_blocking=True),
        }
        raw.update({key: images[:, i] for i, key in enumerate(self.cameras)})
        processed = self.preprocessor(raw)
        processed["valid"] = batch["valid"].to(self.device, non_blocking=True)
        return processed

    def loss(self, batch):
        return self.model(batch)[0]

    def predict(self, batch):
        return self.model.predict_action_chunk(batch)

    def export(self, path, dataset, asset_destination=None):
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        # Local tokenizer/config assets travel with every exported checkpoint.
        if self.assets.resolve() != (path / "vlm_assets").resolve():
            shutil.copytree(
                self.assets, path / "vlm_assets", dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache")
            )
        previous_name = self.model.config.vlm_model_name
        self.model.config.vlm_model_name = str(Path(asset_destination or path / "vlm_assets").resolve())
        try:
            self.model.save_pretrained(path)
        finally:
            self.model.config.vlm_model_name = previous_name
        self.preprocessor.save_pretrained(path, config_filename="policy_preprocessor.json")
        self.postprocessor.save_pretrained(path, config_filename="policy_postprocessor.json")
        write_json(path / "stats.json", read_json(dataset.root / "stats.json"))
        write_json(path / "data_manifest.json", dataset.manifest)


def make_backend(cfg, dataset, device, load_path=None):
    if cfg["backend"] == "tiny_bc":
        return TinyBackend(cfg, dataset, device)
    if cfg["backend"] == "smolvla":
        return SmolBackend(cfg, dataset, device, load_path)
    raise ValueError(f"Unknown backend: {cfg['backend']}")
