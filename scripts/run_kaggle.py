"""Run reproducible Kaggle smoke/full training from a Git checkout.

Execute with the isolated training Python created by setup_kaggle.py.
Data and checkpoints stay in the writable checkout under /kaggle/working.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from robot_foundry.io import read_json, write_json
from robot_foundry.packed import build_packed_cache
from robot_foundry.prepare import prepare


def hardware_report():
    # Probe in a short-lived process: the launcher must not reserve CUDA contexts on both T4s.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from robot_foundry.hardware import probe; print(json.dumps(probe()))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def recipe(args, hardware):
    devices = hardware["gpus"]
    count = min(2, len(devices)) if args.gpus == "auto" else int(args.gpus)
    if count < 1 or len(devices) < count:
        raise RuntimeError(
            f"Requested {args.gpus} GPU(s), but detected {len(devices)}. Enable the Kaggle accelerator."
        )
    if args.backend == "smolvla" and any(g["total_gib"] < 14 for g in devices[:count]):
        raise RuntimeError("SmolVLA profile needs >=14 GiB on each GPU. Two devices do not pool VRAM.")
    if any(g["kernel_error"] for g in devices[:count]):
        raise RuntimeError("CUDA kernel probe failed; inspect doctor and the CUDA wheel.")
    name = args.name or f"{args.backend}_{args.mode}_{count}gpu"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Experiment name may contain only letters, digits, underscores and hyphens.")
    cfg = read_json("configs/kaggle_t4x2.json")
    microbatch = args.microbatch or 1
    if 16 % (microbatch * count):
        raise ValueError("microbatch * GPU count must divide the target effective batch of 16.")
    cfg.update(
        backend=args.backend,
        num_gpus=count,
        batch_size=microbatch,
        gradient_accumulation=16 // (microbatch * count),
        output_dir=f"outputs/{name}",
        prepared_dir=f"data/prepared/{name}",
        precision="fp16" if args.precision == "auto" else args.precision,
        num_workers=args.workers,
    )
    cfg["eval_precision"] = cfg["precision"]
    if args.full_data:
        cfg["episodes_per_task"] = None
    if args.mode == "smoke":
        cfg.update(steps=20, warmup_steps=2, eval_every=10, gradient_accumulation=2, validation_max_samples=8)
    elif args.mode == "benchmark":
        cfg.update(steps=100, warmup_steps=10, eval_every=50, validation_max_samples=8)
    if args.backend == "tiny_bc":
        cfg.update(lr=3e-4)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "benchmark", "full"], default="smoke")
    parser.add_argument("--backend", choices=["smolvla", "tiny_bc"], default="smolvla")
    parser.add_argument("--full-data", action="store_true", help="Use all episodes instead of 10 per task")
    parser.add_argument("--name", default=None, help="Separate name for each experiment")
    parser.add_argument("--resume", action="store_true", help="Continue the same configuration/checkpoint")
    parser.add_argument("--gpus", choices=["auto", "1", "2"], default="auto")
    parser.add_argument("--precision", choices=["auto", "fp16", "fp32"], default="auto")
    parser.add_argument("--microbatch", type=int, choices=[1, 2, 4], default=None)
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers per GPU process")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"Run from the project root: {root}")
    if args.workers < 0:
        raise ValueError("workers must be nonnegative.")
    hardware = hardware_report()
    cfg = recipe(args, hardware)
    cfg_path = Path(cfg["output_dir"]) / "run_config.json"
    manifest_path = Path(cfg["prepared_dir"]) / "manifest.json"
    checkpoint = Path(cfg["output_dir"]) / "training_state.pt"
    if args.resume:
        if not checkpoint.exists() or not cfg_path.exists():
            raise FileNotFoundError("No saved run/config. Restore the previous run and prepared data first.")
        if read_json(cfg_path) != cfg:
            raise ValueError(
                "Resume options differ from saved config; use the same mode/backend/name/full-data flags."
            )
    elif checkpoint.exists():
        raise FileExistsError("Run already exists. Use --resume or a new --name.")
    write_json(cfg_path, cfg)
    result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    write_json(
        Path(cfg["output_dir"]) / "git_provenance.json",
        {"commit": result.stdout.strip(), "python": sys.version.split()[0], "hardware": hardware},
    )
    # Preparation is cache-aware. Changing a data recipe must rebuild its split/stats manifest.
    if not args.resume:
        prepare(cfg_path)
    elif not manifest_path.exists():
        raise FileNotFoundError(
            "Prepared data missing. Restore it together with the checkpoint for reliable resume."
        )
    build_packed_cache(cfg["prepared_dir"])
    if cfg["backend"] == "smolvla":
        # Populate the common Hub cache once before both ranks load the policy.
        from huggingface_hub import snapshot_download

        models = read_json(cfg["registry"])
        snapshot_download(models["smolvla"]["repo_id"], revision=models["smolvla"]["revision"])
        snapshot_download(
            models["smolvlm"]["repo_id"],
            revision=models["smolvlm"]["revision"],
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.onnx"],
            local_dir=str(Path(cfg["output_dir"]).resolve() / "vlm_assets"),
        )
        write_json(Path(cfg["output_dir"]) / "vlm_assets" / "foundry_revision.json", models["smolvlm"])
    print(
        f"Launching {cfg['num_gpus']} GPU process(es), {cfg['precision']}, "
        f"effective batch={cfg['batch_size'] * cfg['gradient_accumulation'] * cfg['num_gpus']}",
        flush=True,
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={cfg['num_gpus']}",
        "-m",
        "robot_foundry.cli",
        "train",
        "--config",
        str(cfg_path),
    ]
    if args.resume:
        command.append("--resume")
    worker_env = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2")
    if cfg["backend"] == "smolvla":
        worker_env["HF_HUB_OFFLINE"] = "1"
    subprocess.run(command, env=worker_env, check=True)
    report = read_json(Path(cfg["output_dir"]) / "report.json")
    print(json.dumps({"output_dir": cfg["output_dir"], "report": report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
