"""Repair disk use and restart/resume the original recipe without changing its fingerprint."""

import argparse
import os
import subprocess
from pathlib import Path

from kaggle_storage import repair
from run_kaggle import hardware_report, launch_training, stage_models

from robot_foundry.io import read_json, write_json
from robot_foundry.storage import GIB, require_free


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--prune-raw", action="store_true")
    parser.add_argument(
        "--train", action="store_true", help="Resume if a durable checkpoint exists; otherwise restart"
    )
    args = parser.parse_args()
    if args.train and not args.apply:
        raise ValueError("--train requires --apply so storage is verified first.")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    result = repair(root, args.run, args.apply, args.prune_raw)
    if not args.train:
        return
    path = Path("outputs") / args.run / "run_config.json"
    cfg = read_json(path)
    devices = hardware_report()["gpus"]
    if len(devices) < cfg["num_gpus"]:
        raise RuntimeError("Recovery requires the original GPU count. Keep the original recipe unchanged.")
    if cfg["backend"] == "smolvla":
        require_free(root, 6 * GIB, "recovering SmolVLA export/checkpoint reserve")
    stage_models(cfg)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(Path(cfg["output_dir"]) / "recovery_provenance.json", {"commit": commit, "storage": result})
    resume = result["resume_checkpoint_exists"]
    print(
        "Resuming durable optimizer checkpoint."
        if resume
        else "No durable optimizer checkpoint: restarting training with prepared data.",
        flush=True,
    )
    launch_training(cfg, path, resume)


if __name__ == "__main__":
    main()
