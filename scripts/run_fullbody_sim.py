"""Run actual SONIC G1 reference inference in MuJoCo and record a full-body video."""

import argparse
from pathlib import Path

from robot_foundry.fullbody import evaluate_sonic, require_g1_vla_checkpoint
from robot_foundry.io import write_json
from robot_foundry.sonic_assets import prepare_sonic_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="vendor/sonic")
    parser.add_argument("--assets", default="data/models/sonic_mujoco")
    parser.add_argument("--motion", default="walking_quip_360_R_002__A428")
    parser.add_argument("--seconds", type=float, default=9.1)
    parser.add_argument("--warmup", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/g1_sonic_fullbody")
    parser.add_argument("--vla-checkpoint", help="Fail explicitly on incompatible Panda/VLA checkpoints")
    args = parser.parse_args()
    if args.vla_checkpoint:
        require_g1_vla_checkpoint(args.vla_checkpoint)
    if Path(args.output).exists():
        raise FileExistsError("Use a new output directory to retain previous evaluations")
    stage = "assets"
    try:
        scene, motion, models, provenance = prepare_sonic_assets(args.repo, args.assets, args.motion)
        stage = "rollout"
        report = evaluate_sonic(scene, motion, models, provenance, args.output, args.seconds, args.warmup)
        write_json(
            Path(args.output) / "simulation_status.json",
            {
                "status": "complete",
                "stage": stage,
                "fall_detected": report["fall_detected"],
            },
        )
    except Exception as error:
        write_json(
            Path(args.output) / "simulation_status.json",
            {
                "status": "failed",
                "stage": stage,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    return 1 if report["fall_detected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
