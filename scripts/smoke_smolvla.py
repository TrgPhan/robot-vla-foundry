"""Validate the real pretrained policy: finite backward, export, then reload prediction.

CPU validation may take several minutes and downloads ~1 GB of public model weights.
Run in the isolated VLA environment, from the project root after local data preparation.
"""

import json
from pathlib import Path

import torch

from robot_foundry.backends import SmolBackend
from robot_foundry.data import EpisodesDataset
from robot_foundry.io import write_json
from robot_foundry.train import seed_everything


def main():
    seed_everything(42)
    cfg = {
        "output_dir": "outputs/smolvla_cpu_smoke",
        "registry": "configs/datasets.json",
        "cpu_smoke_only": True,
        "steps": 1,
    }
    dataset = EpisodesDataset("data/prepared/local", "train", normalize=False)
    backend = SmolBackend(cfg, dataset, torch.device("cpu"))
    # Tail example exercises the real processor's propagation of action padding masks.
    raw = torch.utils.data.default_collate([dataset[dataset.ends[0] - 1]])
    batch = backend.process(raw)
    assert "action_is_pad" in batch and batch["action_is_pad"].any()
    assert torch.equal(batch["action_is_pad"], ~batch["valid"])
    backend.model.train()
    optimizer = torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad], lr=1e-4)
    loss = backend.loss(batch)
    assert torch.isfinite(loss), "Non-finite real SmolVLA loss"
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(backend.model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    target = Path(cfg["output_dir"]) / "checkpoint"
    backend.export(target, dataset)
    del optimizer, backend
    import gc

    gc.collect()
    reloaded = SmolBackend(cfg, dataset, torch.device("cpu"), load_path=target)
    reloaded.model.eval()
    with torch.inference_mode():
        prediction = reloaded.predict(reloaded.process(raw))
    assert prediction.shape == (1, 16, 7) and torch.isfinite(prediction).all()
    report = {
        "pretrained_policy": "lerobot/smolvla_base",
        "real_optimizer_steps": 1,
        "device": "cpu",
        "loss": loss.item(),
        "gradient_norm": float(norm),
        "prediction_shape": list(prediction.shape),
        "export_reload_passed": True,
        "padding_mask_propagated": True,
        "scope": "API integration smoke test; no performance claim",
    }
    write_json("reports/smolvla_smoke.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
