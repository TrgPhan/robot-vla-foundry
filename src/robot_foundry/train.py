"""Single-device training with deterministic sampling, atomic resume, and held-out reports."""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .backends import make_backend
from .data import EpisodesDataset
from .hardware import probe
from .io import fingerprint, read_json, write_json


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_device(cfg):
    name = cfg.get("device", "auto")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch cannot use CUDA. Run doctor.")
    return device


def lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))


@torch.inference_mode()
def evaluate(backend, dataset, batch_size=16, max_batches=None):
    backend.model.eval()
    selected = dataset
    if max_batches is not None and max_batches * batch_size < len(dataset):
        # Spread bounded validation across the entire split, rather than only its first task.
        selected = Subset(
            dataset, np.linspace(0, len(dataset) - 1, max_batches * batch_size, dtype=int).tolist()
        )
    loader = DataLoader(selected, batch_size=batch_size, shuffle=False, num_workers=0)
    std = torch.as_tensor(dataset.action_std, device=backend.device)
    abs_sum = sq_sum = count = first_sum = first_sq = first_count = None
    mean_baseline_sum = 0.0
    batches = 0
    for raw in loader:
        batch = backend.process(raw)
        pred = backend.predict(batch).float()
        target = batch["action"].float()
        valid = batch["valid"].unsqueeze(-1).expand_as(target)
        error = pred - target
        raw_error = error.abs() * std
        this_abs = (raw_error * valid).sum((0, 1))
        this_sq = (error.square() * valid).sum((0, 1))
        this_count = valid.sum((0, 1))
        if abs_sum is None:
            abs_sum, sq_sum, count = this_abs, this_sq, this_count
            first_sum = raw_error[:, 0].sum(0)
            first_sq = error[:, 0].square().sum(0)
            first_count = len(pred)
        else:
            abs_sum += this_abs
            sq_sum += this_sq
            count += this_count
            first_sum += raw_error[:, 0].sum(0)
            first_sq += error[:, 0].square().sum(0)
            first_count += len(pred)
        mean_baseline_sum += (target.square() * valid).sum().item()
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    return {
        "normalized_rmse": torch.sqrt(sq_sum.sum() / count.sum()).item(),
        "mean_action_baseline_normalized_rmse": math.sqrt(mean_baseline_sum / count.sum().item()),
        "raw_mae_per_dimension": (abs_sum / count).cpu().tolist(),
        "first_action_normalized_rmse": torch.sqrt(first_sq.sum() / (first_count * len(std))).item(),
        "first_action_raw_mae_per_dimension": (first_sum / first_count).cpu().tolist(),
        "evaluated_samples": int(first_count),
        "batches": batches,
        "max_batches": max_batches,
        "metric_scope": "open-loop held-out demonstrations; not rollout success or humanoid balance",
    }


def atomic_checkpoint(path, value):
    temp = Path(path).with_suffix(".tmp")
    try:
        torch.save(value, temp)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def train(config_path, resume=False):
    cfg = read_json(config_path)
    if cfg.get("training_engine") == "ddp_amp_v2":
        from .train_optimized import train as optimized_train

        return optimized_train(config_path, resume)
    seed_everything(cfg["seed"])
    if cfg["steps"] <= 0 or cfg["batch_size"] <= 0 or cfg["warmup_steps"] >= cfg["steps"]:
        raise ValueError("Need positive steps/batch and warmup_steps < steps.")
    out = Path(cfg["output_dir"])
    state_path = out / "training_state.pt"
    if out.exists() and state_path.exists() and not resume:
        raise FileExistsError(
            f"Training already exists: {out}. Pass --resume or choose a new output directory."
        )
    if resume and not state_path.exists():
        raise FileNotFoundError(f"No resumable training state: {state_path}")
    out.mkdir(parents=True, exist_ok=True)
    device = to_device(cfg)
    normalized = cfg["backend"] == "tiny_bc"
    ds_train = EpisodesDataset(cfg["prepared_dir"], "train", normalized, cfg.get("cache_episodes", 2))
    ds_val = EpisodesDataset(cfg["prepared_dir"], "val", normalized, cfg.get("cache_episodes", 2))
    ds_test = EpisodesDataset(cfg["prepared_dir"], "test", normalized, cfg.get("cache_episodes", 2))
    if ds_train.manifest["config"]["horizon"] < 1:
        raise ValueError("Invalid prepared data horizon.")
    backend = make_backend(cfg, ds_train, device)
    model = backend.model
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=1e-4
    )
    total = cfg["steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total, cfg["warmup_steps"])
    )
    accumulation = cfg.get("gradient_accumulation", 1)
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be >=1.")
    # Random-with-replacement generator state is checkpointed; resume has the same future sample sequence.
    sample_rng = torch.Generator().manual_seed(cfg["seed"] + 101)
    config_hash = fingerprint(cfg)
    start_step, best = 0, float("inf")
    if resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if (
            state["data_fingerprint"] != ds_train.manifest["fingerprint"]
            or state["config_hash"] != config_hash
        ):
            raise ValueError("Resume configuration/dataset changed. Create a separate experiment.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sample_rng.set_state(state["sample_rng"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step, best = state["step"], state["best"]
        if start_step >= total and (out / "report.json").exists():
            print("Training already complete; returning its existing report.", flush=True)
            return read_json(out / "report.json")
    write_json(out / "config.json", cfg)
    write_json(out / "hardware.json", probe())
    write_json(out / "data_manifest.json", ds_train.manifest)
    wall_start = time.perf_counter()
    train_steps = 0
    print(
        f"Train {cfg['backend']}: device={device}, train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}",
        flush=True,
    )
    for step in range(start_step, total):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(accumulation):
            ids = torch.randint(len(ds_train), (cfg["batch_size"],), generator=sample_rng).tolist()
            raw = torch.utils.data.default_collate([ds_train[i] for i in ids])
            batch = backend.process(raw)
            loss = backend.loss(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}.")
            (loss / accumulation).backward()
            loss_total += loss.detach().item() / accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        train_steps += 1
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == total:
            # Evaluation randomness must not change the next training step (flow sampling/dropout RNG).
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(cfg["seed"] + 303)
                metrics = evaluate(backend, ds_val, cfg["batch_size"], cfg.get("validation_max_batches"))
            improved = metrics["normalized_rmse"] < best
            if improved:
                best = metrics["normalized_rmse"]
                backend.export(out / "best", ds_train)
            row = {
                "step": step + 1,
                "train_loss": loss_total,
                "gradient_norm": float(grad_norm),
                "lr": optimizer.param_groups[0]["lr"],
                "val": metrics,
                "best_val": best,
                "elapsed_seconds": time.perf_counter() - wall_start,
            }
            with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            atomic_checkpoint(
                state_path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "sample_rng": sample_rng.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "step": step + 1,
                    "best": best,
                    "config_hash": config_hash,
                    "data_fingerprint": ds_train.manifest["fingerprint"],
                },
            )
            print(
                f"step={step + 1} loss={loss_total:.4f} val_rmse={metrics['normalized_rmse']:.4f} best={best:.4f}",
                flush=True,
            )
    if cfg["backend"] == "tiny_bc":
        best_state = torch.load(out / "best/model.pt", map_location=device, weights_only=True)
        model.load_state_dict(best_state["model"])
    else:
        # Avoid constructing a second 450M-parameter policy on GPU just to load the selected weights.
        from safetensors.torch import load_file

        model.load_state_dict(load_file(str(out / "best/model.safetensors"), device=str(device)))
    torch.manual_seed(cfg["seed"] + 404)
    report = {
        "backend": cfg["backend"],
        "steps": total,
        "selected_by": "validation normalized RMSE",
        "validation_best_rmse": best,
        "test": evaluate(backend, ds_test, cfg["batch_size"]),
        "wall_seconds_this_invocation": time.perf_counter() - wall_start,
        "steps_this_invocation": train_steps,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "data_fingerprint": ds_train.manifest["fingerprint"],
    }
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report
