"""Compare an interrupted/resumed optimizer trajectory with an uninterrupted one."""

from pathlib import Path

import numpy as np
import pytest
import torch

from robot_foundry.io import fingerprint, write_json
from robot_foundry.train import train


def make_data(root):
    rng = np.random.default_rng(10)
    rows = []
    for split in ["train", "val", "test"]:
        path = root / f"{split}.npz"
        np.savez(
            path,
            images=rng.integers(256, size=(6, 2, 3, 32, 32), dtype=np.uint8),
            states=rng.normal(size=(6, 8)).astype(np.float32),
            actions=rng.normal(size=(6, 4, 7)).astype(np.float32),
            valid=np.ones((6, 4), bool),
            prompts=np.asarray(["pick up milk"] * 6),
        )
        rows.append(
            {
                "source": "synthetic_test_only",
                "episode_index": 0,
                "split": split,
                "samples": 6,
                "path": path.name,
            }
        )
    manifest = {
        "config": {"horizon": 4, "image_size": 32},
        "signature": {"state": {"shape": [8]}, "action": {"shape": [7]}, "cameras": ["image", "wrist"]},
        "episodes": rows,
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(root / "manifest.json", manifest)
    write_json(
        root / "stats.json",
        {"observation.state": {"mean": [0] * 8, "std": [1] * 8}, "action": {"mean": [0] * 7, "std": [1] * 7}},
    )


def test_resume_restores_sampling_optimizer_and_rng(tmp_path, monkeypatch):
    import importlib

    train_module = importlib.import_module("robot_foundry.train")
    root = tmp_path / "data"
    root.mkdir()
    make_data(root)
    cfg = {
        "seed": 42,
        "backend": "tiny_bc",
        "prepared_dir": str(root),
        "device": "cpu",
        "steps": 6,
        "warmup_steps": 1,
        "batch_size": 2,
        "lr": 3e-4,
        "eval_every": 2,
        "output_dir": str(tmp_path / "full"),
    }
    full_config = tmp_path / "full.json"
    write_json(full_config, cfg)
    train(full_config)
    cfg["output_dir"] = str(tmp_path / "resumed")
    resume_config = tmp_path / "resume.json"
    write_json(resume_config, cfg)
    real_save = train_module.atomic_checkpoint

    def interrupt(path, state):
        real_save(path, state)
        if state["step"] == 2:
            raise InterruptedError("Simulated session interruption after an atomic save")

    with monkeypatch.context() as context:
        context.setattr(train_module, "atomic_checkpoint", interrupt)
        with pytest.raises(InterruptedError):
            train(resume_config)
    train(resume_config, resume=True)
    a = torch.load(tmp_path / "full/training_state.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "resumed/training_state.pt", map_location="cpu", weights_only=False)
    assert a["step"] == b["step"] == 6
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert a["best"] == b["best"]
    torch.testing.assert_close(a["sample_rng"], b["sample_rng"], rtol=0, atol=0)
    # A completed resume must not overwrite the original timing/VRAM report.
    report_path = Path(cfg["output_dir"]) / "report.json"
    original = report_path.read_bytes()
    train(resume_config, resume=True)
    assert report_path.read_bytes() == original
