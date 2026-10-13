"""Explicit, inspectable command-line workflows."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(prog="robot-foundry")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("doctor")
    p.add_argument("--output", default="outputs/hardware.json")
    p = sub.add_parser("audit-sources")
    p.add_argument("--registry", default="configs/datasets.json")
    p.add_argument("--output", default="outputs/source_audit.json")
    for name in ["prepare", "train"]:
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/local.json")
        if name == "train":
            p.add_argument("--resume", action="store_true")
    p = sub.add_parser("bootstrap")
    p.add_argument("--repos", nargs="+", choices=["groot", "sonic", "lerobot", "openpi"], required=True)
    p.add_argument("--vendor", default="vendor")
    p = sub.add_parser("export-groot")
    p.add_argument("--prepared", default="data/prepared/local")
    p.add_argument("--output", default="data/native/libero")
    p.add_argument("--groot-repo", default="vendor/groot")
    p.add_argument("--keep-av1", action="store_true")
    p = sub.add_parser("train-groot")
    p.add_argument("--config", default="configs/groot_libero.json")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("validate-g1")
    p.add_argument("--dataset", required=True)
    p = sub.add_parser("eval-libero")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backend", choices=["tiny_bc", "smolvla"], default="tiny_bc")
    p.add_argument(
        "--suite",
        choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"],
        default="libero_object",
    )
    p.add_argument("--episodes-per-task", type=int, default=50)
    p.add_argument("--execution-horizon", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--policy-url", default=None)
    p.add_argument("--output", default="outputs/libero_eval")
    p.add_argument("--record-video", action="store_true", help="Save both policy cameras as MP4 per trial")
    args = parser.parse_args()
    if args.command == "doctor":
        from .hardware import probe
        from .io import write_json

        report = probe()
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "audit-sources":
        from .datasets import audit_registry

        print(json.dumps(audit_registry(args.registry, args.output), indent=2))
    elif args.command == "prepare":
        from .prepare import prepare

        prepare(args.config)
    elif args.command == "train":
        from .train import train

        train(args.config, args.resume)
    elif args.command == "bootstrap":
        from .native import bootstrap

        bootstrap(args.repos, args.vendor)
    elif args.command == "export-groot":
        from .native import export_native

        print(json.dumps(export_native(args.prepared, args.output, args.groot_repo, not args.keep_av1)))
    elif args.command == "train-groot":
        from .native import launch_groot

        launch_groot(args.config, args.dry_run)
    elif args.command == "validate-g1":
        from .native import validate_g1_dataset

        print(json.dumps(validate_g1_dataset(args.dataset), indent=2))
    elif args.command == "eval-libero":
        from .simulation import evaluate_libero

        evaluate_libero(args)


if __name__ == "__main__":
    main()
