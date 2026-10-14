"""Train a G1 latent policy on paired simulation demonstrations, with DDP/AMP and durable checkpoints."""

import argparse
import os
import re
import subprocess
from pathlib import Path

from robot_foundry.g1_schema import ACTION_CONVENTION, validate_g1_manifest
from robot_foundry.io import read_json, write_json
from robot_foundry.packed import validate_cache
from run_kaggle import hardware_report, launch_training, stage_models


def recipe(args, hardware):
    cfg = read_json("configs/g1_kaggle.json")
    count = min(2, len(hardware["gpus"])) if args.gpus == "auto" else int(args.gpus)
    if count < 1 or len(hardware["gpus"]) < count:
        raise RuntimeError("Enable GPU acceleration; this launcher runs one process per available GPU")
    if args.backend == "smolvla" and any(g["total_gib"] < 14 for g in hardware["gpus"][:count]):
        raise RuntimeError(
            "Production SmolVLA G1 profile needs >=14 GiB per GPU; local diagnostics are separate"
        )
    if any(g["kernel_error"] for g in hardware["gpus"][:count]):
        raise RuntimeError("CUDA kernel probe failed")
    if 16 % (args.microbatch * count):
        raise ValueError("microbatch * GPU count must divide effective batch 16")
    cfg.update(
        backend=args.backend,
        output_dir=f"outputs/{args.name}",
        prepared_dir=args.prepared,
        num_gpus=count,
        batch_size=args.microbatch,
        num_workers=args.workers,
        gradient_accumulation=16 // (args.microbatch * count),
    )
    if args.backend == "tiny_bc":
        cfg["lr"] = 3e-4
    if args.mode == "smoke":
        cfg.update(
            steps=20,
            warmup_steps=2,
            eval_every=10,
            gradient_accumulation=2,
            validation_max_samples=8,
            test_max_samples=8,
        )
    if args.steps is not None:
        if args.steps < 1:
            raise ValueError("steps must be positive")
        cfg.update(
            steps=args.steps,
            warmup_steps=min(cfg["warmup_steps"], args.steps // 20),
            eval_every=min(cfg["eval_every"], args.steps),
        )
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="g1_smolvla_full")
    p.add_argument("--prepared", default="data/prepared/g1_vla_v1")
    p.add_argument("--backend", choices=["smolvla", "tiny_bc"], default="smolvla")
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--gpus", choices=["auto", "1", "2"], default="auto")
    p.add_argument("--microbatch", type=int, choices=[1, 2, 4], default=1)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--steps", type=int)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name) or args.workers < 0:
        raise ValueError("Use a simple run name and nonnegative workers")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    manifest = read_json(Path(args.prepared) / "manifest.json")
    validate_g1_manifest(manifest)
    for record in manifest["episodes"]:
        validate_cache(args.prepared, record)
    cfg = recipe(args, hardware_report())
    if cfg["action_convention"] != ACTION_CONVENTION:
        raise ValueError("G1 training recipe action space changed unexpectedly")
    config_path = Path(cfg["output_dir"]) / "run_config.json"
    if config_path.exists():
        if not args.resume or read_json(config_path) != cfg:
            raise ValueError("Existing run: resume with exactly the same options, or choose a new name")
    else:
        if args.resume:
            raise FileNotFoundError("No saved run configuration to resume")
        write_json(config_path, cfg)
    write_json(
        Path(cfg["output_dir"]) / "git_provenance.json",
        {
            "task": manifest["scope"],
            "data_fingerprint": manifest["fingerprint"],
            "action_dimension": 64,
            "state_dimension": 70,
            "controller_identity": manifest["controller_identity"],
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            ),
        },
    )
    stage_models(cfg)
    launch_training(cfg, config_path, args.resume)


if __name__ == "__main__":
    main()
