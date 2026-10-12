"""Prepare official SONIC motion evaluation/training commands; pretrained controller is the default.

Run --dry-run on any machine. --run needs the SONIC/Isaac Lab Python environment on Linux.
This does not convert LIBERO actions into humanoid labels or add a pretend balance loss.
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from robot_foundry.io import read_json, write_json
from robot_foundry.native import check_checkout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="vendor/sonic")
    parser.add_argument("--motion-dir", required=True, help="Filtered G1 motion_lib PKLs, not raw CSVs")
    parser.add_argument("--mode", choices=["eval", "finetune"], default="eval")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--smpl-motion-dir", default=None)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    check_checkout("sonic", repo)
    if args.num_envs < 1:
        raise ValueError("num-envs must be positive")
    model = read_json("configs/datasets.json")["sonic"]
    checkpoint = (
        Path(args.checkpoint).resolve() if args.checkpoint else repo / "checkpoints/sonic_v1_1/last.pt"
    )
    motion = str(Path(args.motion_dir).resolve())
    if args.mode == "eval":
        command = [
            sys.executable,
            "gear_sonic/eval_agent_trl.py",
            f"+checkpoint={checkpoint}",
            "+headless=True",
            "++eval_callbacks=im_eval",
            "++run_eval_loop=False",
            f"++num_envs={args.num_envs}",
            "+manager_env/terminations=tracking/eval",
            f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion}",
            "++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=512",
        ]
    else:
        if not args.smpl_motion_dir:
            raise ValueError(
                "Finetuning needs paired SMPL motion PKLs for the released three-encoder config."
            )
        command = [
            sys.executable,
            "gear_sonic/train_agent_trl.py",
            "+exp=manager/universal_token/all_modes/sonic_v1_1",
            f"+checkpoint={checkpoint}",
            f"num_envs={args.num_envs}",
            "headless=True",
            f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion}",
            "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
            + str(Path(args.smpl_motion_dir).resolve()),
        ]
    plan = {
        "cwd": str(repo),
        "argv": command,
        "model_repo": model["repo_id"],
        "model_revision": model["revision"],
        "variant": "sonic_v1_1",
        "mode": args.mode,
        "scope": "motion tracking only; does not benchmark visual manipulation or object dynamics",
    }
    print(json.dumps(plan, indent=2))
    if args.run:
        if platform.system() != "Linux":
            raise RuntimeError(
                "Official SONIC/Isaac Lab execution targets the separately installed Linux environment."
            )
        if not Path(motion).is_dir():
            raise FileNotFoundError(motion)
        if not args.checkpoint:
            for file in ["last.pt", "config.yaml", "model_config.yaml"]:
                hf_hub_download(
                    model["repo_id"],
                    f"sonic_v1_1/{file}",
                    revision=model["revision"],
                    local_dir=str(repo / "checkpoints"),
                )
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        write_json("outputs/sonic_launch_plan.json", plan)
        subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
