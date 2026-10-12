"""DDP/AMP training with disjoint data, distributed metrics and per-rank resume state."""

import hashlib
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from .backends import make_backend
from .data import EpisodesDataset, masked_loss
from .distributed import Runtime, StepBatchSampler
from .io import fingerprint, read_json, write_json
from .precision import autocast
from .train import atomic_checkpoint, lr_factor, seed_everything


class LossModule(nn.Module):
    def __init__(self, policy, cfg):
        super().__init__()
        self.policy = policy
        self.backend = cfg["backend"]
        self.velocity_weight = cfg.get("velocity_weight", 0.05)

    def forward(self, batch):
        if self.backend == "tiny_bc":
            return masked_loss(
                self.policy(batch).float(), batch["action"].float(), batch["valid"], self.velocity_weight
            )
        return self.policy(batch)[0].float()


def loader_options(cfg, device, workers=None):
    count = cfg.get("num_workers", 0) if workers is None else workers
    options = {
        "num_workers": count,
        "pin_memory": device.type == "cuda",
        "generator": torch.Generator().manual_seed(cfg["seed"] + 505),
    }
    if count:
        options.update(
            persistent_workers=True,
            prefetch_factor=cfg.get("prefetch_factor", 2),
            multiprocessing_context="spawn",
        )
    return options


@torch.inference_mode()
def evaluate(backend, dataset, cfg, runtime, split="val"):
    """Sum sufficient statistics: no padded validation duplicates or averaging per-rank RMSE."""
    backend.model.eval()
    cap = cfg.get("validation_max_samples") if split == "val" else cfg.get("test_max_samples")
    indices = np.arange(len(dataset))
    if cap is not None and cap < len(indices):
        indices = np.linspace(0, len(dataset) - 1, cap, dtype=np.int64)
    local = indices[runtime.rank :: runtime.world_size].tolist()
    loader = DataLoader(
        Subset(dataset, local),
        batch_size=cfg["batch_size"],
        shuffle=False,
        **loader_options(cfg, runtime.device, workers=cfg.get("eval_num_workers", 0)),
    )
    dim = len(dataset.action_std)
    totals = torch.zeros(6 * dim + 3, dtype=torch.float64, device=runtime.device)
    std = torch.as_tensor(dataset.action_std, device=runtime.device)
    precision = cfg.get("eval_precision", cfg.get("precision", "fp32"))
    failed = False
    for raw in loader:
        batch = backend.process(raw)
        with autocast(runtime.device, precision):
            prediction = backend.predict(batch).float()
        failed = failed or not torch.isfinite(prediction).all().item()
        target = batch["action"].float()
        error = (prediction - target).double()
        mask = batch["valid"].unsqueeze(-1).expand_as(error)
        absolute = error.abs() * std
        values = [
            (absolute * mask).sum((0, 1)),
            (error.square() * mask).sum((0, 1)),
            mask.sum((0, 1)),
            absolute[:, 0].sum(0),
            error[:, 0].square().sum(0),
            torch.full((dim,), len(prediction), device=runtime.device),
        ]
        for i, value in enumerate(values):
            totals[i * dim : (i + 1) * dim] += value
        totals[-3] += (target.double().square() * mask).sum()
        totals[-2] += len(prediction)
        totals[-1] += 1
    if not runtime.all_true(not failed):
        raise FloatingPointError(f"Non-finite {split} prediction.")
    runtime.sum(totals)
    absolute, square, counts, first_absolute, first_square, first_counts = totals[:-3].reshape(6, dim)
    if totals[-2] == 0:
        raise ValueError("Empty evaluation split.")
    return {
        "normalized_rmse": torch.sqrt(square.sum() / counts.sum()).item(),
        "mean_action_baseline_normalized_rmse": torch.sqrt(totals[-3] / counts.sum()).item(),
        "raw_mae_per_dimension": (absolute / counts).cpu().tolist(),
        "first_action_normalized_rmse": torch.sqrt(first_square.sum() / first_counts.sum()).item(),
        "first_action_raw_mae_per_dimension": (first_absolute / first_counts).cpu().tolist(),
        "evaluated_samples": int(totals[-2].item()),
        "batches_across_ranks": int(totals[-1].item()),
        "sample_cap": cap,
        "precision": precision,
        "metric_scope": "held-out open-loop demonstrations; not rollout success or balance",
    }


def rng_state(runtime):
    return {
        "rank": runtime.rank,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state(runtime.device) if runtime.device.type == "cuda" else None,
    }


def restore_rng(state, runtime):
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if runtime.device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], runtime.device)


def parameter_digest(model):
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def train(config_path, resume=False):
    cfg = read_json(config_path)
    if cfg["steps"] < 1 or cfg["batch_size"] < 1 or not 0 <= cfg["warmup_steps"] < cfg["steps"]:
        raise ValueError("Invalid steps/batch/warmup configuration.")
    if cfg["backend"] == "smolvla":
        # Load optional policy libraries before starting distributed backend threads.
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        del make_pre_post_processors, SmolVLAPolicy
    runtime = Runtime(cfg)
    try:
        return run_training(cfg, runtime, resume)
    finally:
        runtime.close()


def run_training(cfg, runtime, resume):
    if cfg.get("num_gpus", runtime.world_size) != runtime.world_size:
        raise ValueError("Configured GPU count differs from launched process count; use run_kaggle.py.")
    for key in ["gradient_accumulation", "eval_every", "validation_max_samples", "test_max_samples"]:
        if cfg.get(key) is not None and cfg[key] < 1:
            raise ValueError(f"{key} must be positive.")
    device = runtime.device
    precision = cfg.get("precision", "fp32")
    with autocast(device, precision):
        pass
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg.get("cpu_threads_per_rank", 2))
    out = Path(cfg["output_dir"])
    state_path = out / "training_state.pt"
    if state_path.exists() and not resume:
        raise FileExistsError(f"Existing run {out}; use --resume or a fresh name.")
    if resume and not state_path.exists():
        raise FileNotFoundError(state_path)
    runtime.primary_call(lambda: out.mkdir(parents=True, exist_ok=True))
    normalized = cfg["backend"] == "tiny_bc"
    datasets = {
        s: EpisodesDataset(
            cfg["prepared_dir"],
            s,
            normalized,
            cfg.get("cache_episodes", 4),
            packed=cfg.get("packed_cache", False),
        )
        for s in ["train", "val", "test"]
    }
    dataset = datasets["train"]
    backend = make_backend(cfg, dataset, device)
    model = backend.model
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-4),
        fused=device.type == "cuda" and cfg.get("fused_optimizer", True),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, cfg["steps"], cfg["warmup_steps"])
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision == "fp16",
        init_scale=cfg.get("amp_init_scale", 1024),
    )
    accumulation = cfg.get("gradient_accumulation", 1)
    config_hash = fingerprint(cfg)
    start, best, successful, skipped = 0, float("inf"), 0, 0
    consecutive_skips = 0
    resume_rng = None
    if resume:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("engine") != "ddp_amp_v2" or state["world_size"] != runtime.world_size:
            raise ValueError("Resume requires the same engine/world size; start fresh for topology changes.")
        if (
            state["config_hash"] != config_hash
            or state["data_fingerprint"] != dataset.manifest["fingerprint"]
        ):
            raise ValueError("Resume configuration/dataset changed.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best = state["step"], state["best"]
        successful, skipped = state["optimizer_updates"], state["amp_skipped_updates"]
        consecutive_skips = state.get("consecutive_amp_skips", 0)
        resume_rng = state["rank_states"][runtime.rank]
        del state
        if start >= cfg["steps"] and (out / "report.json").exists():
            return read_json(out / "report.json")
    loss_module = LossModule(model, cfg)
    if runtime.distributed:
        loss_module = DistributedDataParallel(
            loss_module,
            device_ids=[runtime.local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=cfg.get("ddp_find_unused_parameters", cfg["backend"] == "smolvla"),
        )
    seed_everything(cfg["seed"] + runtime.rank * 997)
    torch.set_num_threads(cfg.get("cpu_threads_per_rank", 2))
    if resume_rng:
        restore_rng(resume_rng, runtime)
    sampler = StepBatchSampler(
        len(dataset),
        cfg["batch_size"],
        accumulation,
        cfg["steps"],
        cfg["seed"],
        runtime.rank,
        runtime.world_size,
        start,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, **loader_options(cfg, device))
    batches = iter(loader)
    runtime.primary_call(lambda: write_json(out / "config.json", cfg))
    runtime.primary_call(lambda: write_json(out / "data_manifest.json", dataset.manifest))
    effective = cfg["batch_size"] * accumulation * runtime.world_size
    print(
        f"rank={runtime.rank}/{runtime.world_size} device={device} precision={precision} "
        f"microbatch={cfg['batch_size']} accumulation={accumulation} effective_batch={effective}",
        flush=True,
    )
    wall_start = interval_start = time.perf_counter()
    interval_steps = 0
    for step in range(start, cfg["steps"]):
        loss_module.train()
        optimizer.zero_grad(set_to_none=True)
        local_loss = torch.zeros((), device=device)
        for micro in range(accumulation):
            batch = backend.process(next(batches))
            sync = (
                loss_module.no_sync() if runtime.distributed and micro + 1 < accumulation else nullcontext()
            )
            with sync:
                with autocast(device, precision):
                    loss = loss_module(batch)
                scaler.scale(loss / accumulation).backward()
            local_loss += loss.detach() / accumulation
        if not runtime.all_true(torch.isfinite(local_loss).item()):
            raise FloatingPointError(f"Non-finite loss at step {step + 1}.")
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(
            trainable, cfg.get("grad_clip_norm", 1.0), error_if_nonfinite=False, foreach=True
        )
        finite = runtime.all_true(torch.isfinite(norm).item())
        if finite:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            successful += 1
            consecutive_skips = 0
        elif scaler.is_enabled():
            scaler.update(new_scale=scaler.get_scale() / 2)
            skipped += 1
            consecutive_skips += 1
            if consecutive_skips > cfg.get("max_consecutive_amp_skips", 10):
                raise FloatingPointError(
                    "Repeated FP16 overflow; inspect data or start a fresh FP32 experiment."
                )
        else:
            raise FloatingPointError("Non-finite FP32 gradients.")
        interval_steps += 1
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps"]:
            loss_mean = runtime.sum(local_loss.clone()) / runtime.world_size
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times = runtime.gather(time.perf_counter() - interval_start)
            with torch.random.fork_rng(devices=[runtime.local_rank] if device.type == "cuda" else []):
                torch.manual_seed(cfg["seed"] + 303 + runtime.rank)
                metrics = evaluate(backend, datasets["val"], cfg, runtime)
            if metrics["normalized_rmse"] < best:
                best = metrics["normalized_rmse"]
                runtime.primary_call(lambda: backend.export(out / "best", dataset))
            ranks = runtime.gather(rng_state(runtime))
            memory = runtime.gather(
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            )
            row = {
                "step": step + 1,
                "train_loss": loss_mean.item(),
                "gradient_norm": float(norm) if finite else None,
                "lr": optimizer.param_groups[0]["lr"],
                "val": metrics,
                "best_val": best,
                "optimizer_updates": successful,
                "amp_skipped_updates": skipped,
                "grad_scale": scaler.get_scale(),
                "train_samples_per_second": interval_steps * effective / max(times),
                "train_seconds_per_step": max(times) / interval_steps,
                "world_size": runtime.world_size,
                "effective_batch": effective,
                "peak_cuda_memory_mib_by_rank": memory,
            }

            def save():
                with (out / "metrics.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, allow_nan=False) + "\n")
                atomic_checkpoint(
                    state_path,
                    {
                        "engine": "ddp_amp_v2",
                        "world_size": runtime.world_size,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "rank_states": ranks,
                        "step": step + 1,
                        "best": best,
                        "optimizer_updates": successful,
                        "amp_skipped_updates": skipped,
                        "consecutive_amp_skips": consecutive_skips,
                        "config_hash": config_hash,
                        "data_fingerprint": dataset.manifest["fingerprint"],
                    },
                )

            runtime.primary_call(save)
            if runtime.primary:
                print(
                    f"step={step + 1} loss={row['train_loss']:.4f} val_rmse={metrics['normalized_rmse']:.4f} "
                    f"samples/s={row['train_samples_per_second']:.2f} amp_skips={skipped}",
                    flush=True,
                )
            interval_steps = 0
            interval_start = time.perf_counter()
    training_checksums = runtime.gather(parameter_digest(model)) if cfg.get("verify_ddp_sync") else None
    if cfg["backend"] == "tiny_bc":
        selected = torch.load(out / "best/model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(selected["model"])
        del selected
    else:
        from safetensors.torch import load_file

        selected = load_file(str(out / "best/model.safetensors"), device="cpu")
        model.load_state_dict(selected)
        del selected
    torch.manual_seed(cfg["seed"] + 404 + runtime.rank)
    test = evaluate(backend, datasets["test"], cfg, runtime, split="test")
    checksums = runtime.gather(parameter_digest(model)) if cfg.get("verify_ddp_sync") else None
    memory = runtime.gather(
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    )
    report = {
        "backend": cfg["backend"],
        "engine": "ddp_amp_v2",
        "world_size": runtime.world_size,
        "precision": precision,
        "effective_batch": effective,
        "steps": cfg["steps"],
        "optimizer_updates": successful,
        "amp_skipped_updates": skipped,
        "validation_best_rmse": best,
        "test": test,
        "selected_by": "validation normalized RMSE",
        "wall_seconds_this_invocation": time.perf_counter() - wall_start,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "peak_cuda_memory_mib_by_rank": memory,
        "parameter_digests_by_rank": checksums,
        "training_parameter_digests_by_rank": training_checksums,
        "data_fingerprint": dataset.manifest["fingerprint"],
    }
    runtime.primary_call(lambda: write_json(out / "report.json", report))
    if runtime.primary:
        print(json.dumps(report, indent=2), flush=True)
    return report
