"""Evaluate a saved run without retraining; report optional simulation failures explicitly."""

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from robot_foundry.io import read_json, write_json
from robot_foundry.sim_environment import simulation_environment


def stop_server(server):
    if server is None or server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=15)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=15)


def evaluate(
    project,
    run_name,
    trials=1,
    suite="libero_object",
    skip_setup=False,
    sim_env=".venv-sim-cpu",
    record_video=False,
):
    project = Path(project).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Invalid run name")
    out = project / "outputs" / run_name
    if out.is_symlink() or out.resolve().parent != (project / "outputs").resolve():
        raise ValueError("Run must stay inside project outputs")
    if trials < 1 or trials > 50:
        raise ValueError("Use 1 to 50 unique LIBERO initial states per task")
    if suite not in ["libero_object", "libero_spatial", "libero_goal", "libero_10"]:
        raise ValueError("Unsupported LIBERO suite")
    cfg = read_json(out / "run_config.json")
    checkpoint = out / "best"
    status_path = out / "simulation_status.json"
    evaluation_out = project / "outputs" / f"libero_{run_name}_{suite}_{trials}"
    sim_root = (project / sim_env).resolve()
    if not sim_root.is_relative_to(project) or sim_root in [project, Path(sys.prefix).resolve()]:
        raise ValueError("Use a separate simulation environment inside the project")
    sim_python = sim_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    report = {
        "status": "running",
        "stage": "preflight",
        "training_report_present": (out / "report.json").is_file(),
        "checkpoint": str(checkpoint),
        "suite": suite,
        "trials_per_task": trials,
        "simulation_output": str(evaluation_out),
        "simulation_torch_index": "cpu",
    }
    server = log = None
    worker_env = simulation_environment()

    def publish():
        try:
            write_json(status_path, report)
        except OSError as error:
            print(f"Could not write simulation status: {error}", flush=True)

    def run(argv):
        subprocess.run([str(x) for x in argv], cwd=project, check=True, env=worker_env)

    try:
        publish()
        read_json(checkpoint / "data_manifest.json")
        model = checkpoint / ("model.pt" if cfg["backend"] == "tiny_bc" else "model.safetensors")
        if not model.is_file():
            raise FileNotFoundError(f"No exported policy: {model}")
        if evaluation_out.exists():
            raise FileExistsError(f"Evaluation directory already exists; retain or move it: {evaluation_out}")
        if not skip_setup:
            report["stage"] = "setup"
            publish()
            run(
                [
                    sys.executable,
                    "scripts/setup_sim.py",
                    "--env",
                    sim_root,
                    "--python",
                    "3.12",
                    "--torch-index",
                    "cpu",
                ]
            )
        if not sim_python.is_file():
            raise FileNotFoundError(f"Simulation Python not found: {sim_python}")
        report["stage"] = "policy_server"
        publish()
        # Never connect to an unrelated process already occupying the policy port.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 8765))
        log = (out / "policy_server.log").open("w", encoding="utf-8")
        precision = cfg.get("precision", "fp16") if cfg["backend"] == "smolvla" else "fp32"
        server = subprocess.Popen(
            [
                sys.executable,
                "scripts/policy_server.py",
                "--checkpoint",
                str(checkpoint),
                "--backend",
                cfg["backend"],
                "--precision",
                precision,
                "--port",
                "8765",
                "--seed",
                str(cfg.get("seed", 42)),
            ],
            cwd=project,
            env=worker_env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        run([sys.executable, "scripts/wait_policy_server.py", "--port", "8765"])
        if server.poll() is not None:
            raise RuntimeError("Owned policy server exited; inspect the run's policy_server.log")
        report["stage"] = "rollout"
        publish()
        rollout = [
            sim_python,
            "-m",
            "robot_foundry.cli",
            "eval-libero",
            "--checkpoint",
            checkpoint,
            "--backend",
            cfg["backend"],
            "--suite",
            suite,
            "--episodes-per-task",
            trials,
            "--max-steps",
            520 if suite == "libero_10" else 300,
            "--execution-horizon",
            4,
            "--seed",
            cfg.get("seed", 42),
            "--output",
            evaluation_out,
            "--policy-url",
            "http://127.0.0.1:8765",
        ]
        if record_video:
            rollout += ["--record-video"]
        run(rollout)
        result = read_json(evaluation_out / "report.json")
        report.update(status="complete", stage="complete", rollout_report=result)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        print(f"Optional simulation failed at {report['stage']}: {error}", flush=True)
        print(f"Saved training artifacts remain at {out}. No training command was run.", flush=True)
    finally:
        try:
            stop_server(server)
        except Exception as error:
            report.update(status="failed", cleanup_error=str(error))
        if log is not None:
            log.close()
        report["free_disk_gib_after"] = shutil.disk_usage(project).free / 2**30
        publish()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--suite",
        choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"],
        default="libero_object",
    )
    parser.add_argument("--skip-setup", action="store_true", help="Reuse an already installed CPU sim env")
    parser.add_argument("--sim-env", default=".venv-sim-cpu")
    parser.add_argument("--record-video", action="store_true")
    args = parser.parse_args()
    report = evaluate(
        Path(__file__).resolve().parents[1],
        args.run,
        args.trials,
        args.suite,
        args.skip_setup,
        args.sim_env,
        args.record_video,
    )
    print(f"Simulation status: {report['status']}", flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
