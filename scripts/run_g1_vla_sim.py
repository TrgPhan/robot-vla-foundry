"""Start an owned trained-policy server, run G1 dynamics/video, and always stop the server."""

import argparse
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

from robot_foundry.io import read_json, write_json
from robot_foundry.sim_environment import simulation_environment
from run_kaggle_sim import stop_server


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--sim-env", default=".venv-fullbody")
    p.add_argument("--output")
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--skip-setup", action="store_true")
    p.add_argument("--policy-mode", choices=["trained", "mean_token"], default="trained")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run):
        raise ValueError("Invalid run name")
    project = Path(__file__).resolve().parents[1]
    run = project / "outputs" / args.run
    cfg = read_json(run / "run_config.json")
    sim_root = (project / args.sim_env).resolve()
    if not sim_root.is_relative_to(project) or sim_root == project or sim_root == Path(sys.prefix).resolve():
        raise ValueError("Use a separate simulator environment inside the project")
    sim_python = sim_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    output = Path(args.output) if args.output else project / "outputs" / f"g1_eval_{args.run}_{args.trials}"
    status_path = run / (
        "g1_simulation_status.json" if args.policy_mode == "trained" else "g1_mean_baseline_status.json"
    )
    status = {"status": "running", "stage": "setup", "policy_mode": args.policy_mode, "output": str(output)}
    server = None
    child_env = simulation_environment()
    try:
        write_json(status_path, status)
        if not args.skip_setup:
            subprocess.run(
                [sys.executable, "scripts/setup_fullbody_sim.py", "--env", str(sim_root)],
                cwd=project,
                env=child_env,
                check=True,
            )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 8775))
        status["stage"] = "trained_policy_server"
        write_json(status_path, status)
        with (run / "g1_policy_server.log").open("w", encoding="utf-8") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "scripts/g1_policy_server.py",
                    "--checkpoint",
                    str(run / "best"),
                    "--backend",
                    cfg["backend"],
                    "--seed",
                    str(cfg["seed"]),
                ],
                cwd=project,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            subprocess.run(
                [sys.executable, "scripts/wait_policy_server.py", "--port", "8775"],
                cwd=project,
                env=child_env,
                check=True,
            )
            if server.poll() is not None:
                raise RuntimeError("Owned G1 policy server exited before rollout")
            status["stage"] = "learned_fullbody_rollout"
            write_json(status_path, status)
            command = [
                str(sim_python),
                "scripts/eval_g1_policy.py",
                "--checkpoint",
                str(run / "best"),
                "--output",
                str(output),
                "--trials",
                str(args.trials),
            ]
            if args.max_frames is not None:
                command += ["--max-frames", str(args.max_frames)]
            command += ["--policy-mode", args.policy_mode]
            subprocess.run(command, cwd=project, env=child_env, check=True)
        result = read_json(output / "report.json")
        status.update(status="complete", stage="complete", rollout_report=result)
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        stop_server(server)
        write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
