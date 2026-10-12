"""Run reproducible Kaggle smoke/full training from a Git checkout.

Execute with the isolated training Python created by setup_kaggle.py.
Data and checkpoints stay in the writable checkout under /kaggle/working.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from robot_foundry.hardware import probe, require_gpu
from robot_foundry.io import read_json, write_json
from robot_foundry.prepare import prepare
from robot_foundry.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--backend", choices=["smolvla", "tiny_bc"], default="smolvla")
    parser.add_argument("--full-data", action="store_true", help="Use all episodes instead of 10 per task")
    parser.add_argument("--name", default=None, help="Separate name for each experiment")
    parser.add_argument("--resume", action="store_true", help="Continue the same configuration/checkpoint")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"Run from the project root: {root}")
    name = args.name or f"{args.backend}_{args.mode}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Experiment name may contain only letters, digits, underscores and hyphens.")
    cfg = read_json("configs/kaggle.json")
    cfg["backend"] = args.backend
    cfg["output_dir"] = f"outputs/{name}"
    cfg["prepared_dir"] = f"data/prepared/{name}"
    if args.full_data:
        cfg["episodes_per_task"] = None
    if args.mode == "smoke":
        cfg.update(steps=20, warmup_steps=2, eval_every=10, gradient_accumulation=2, validation_max_batches=4)
    if args.backend == "smolvla":
        require_gpu("smolvla")  # Check before downloading/decoding datasets.
    else:
        cfg.update(batch_size=16, gradient_accumulation=1, cache_episodes=16, lr=3e-4, device="auto")
    cfg_path = Path("outputs") / name / "run_config.json"
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
        {"commit": result.stdout.strip(), "python": sys.version.split()[0], "hardware": probe()},
    )
    # Preparation is cache-aware. Changing a data recipe must rebuild its split/stats manifest.
    if not args.resume:
        prepare(cfg_path)
    elif not manifest_path.exists():
        raise FileNotFoundError(
            "Prepared data missing. Restore it together with the checkpoint for reliable resume."
        )
    report = train(cfg_path, resume=args.resume)
    print(json.dumps({"output_dir": cfg["output_dir"], "report": report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
