"""Install an isolated CPU SONIC/MuJoCo runtime; leaves training environments untouched."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".venv-fullbody")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    env_path = project / args.env
    if (
        env_path.is_symlink()
        or not env_path.resolve().is_relative_to(project)
        or env_path.resolve() == project
    ):
        raise ValueError("Use a separate environment directory inside the project")
    if not shutil.which("uv"):
        raise RuntimeError("Install uv first: python -m pip install uv==0.10.11")
    if not shutil.which("git"):
        raise RuntimeError("Install Git first to retrieve the pinned SONIC source/assets")
    if shutil.disk_usage(project).free < 2 * 2**30:
        raise OSError(28, "Reserve at least 2 GiB for the CPU full-body environment and assets")

    def run(args):
        subprocess.run([str(x) for x in args], cwd=project, check=True)

    if not (env_path / "pyvenv.cfg").exists():
        if env_path.exists() and any(env_path.iterdir()):
            raise FileExistsError("Environment destination is nonempty and not a virtual environment")
        run(["uv", "venv", "--python", "3.12", env_path])
    executable = env_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(["uv", "pip", "install", "--no-cache", "--python", executable, "-r", "requirements/fullbody_sim.txt"])
    # Install only this runtime profile, rather than pulling the package's training dependencies.
    run(["uv", "pip", "install", "--no-cache", "--python", executable, "--no-deps", "-e", "."])
    run([executable, "-c", "from robot_foundry.upstream import bootstrap; bootstrap(['sonic'])"])
    run(
        [
            executable,
            "-c",
            "import sys, numpy, mujoco, onnxruntime; "
            "from robot_foundry.fullbody import SonicController; "
            "assert 'torch' not in sys.modules; "
            "print('CPU full-body runtime:', numpy.__version__, mujoco.__version__, onnxruntime.__version__)",
        ]
    )
    print(f"Full-body Python: {executable}", flush=True)


if __name__ == "__main__":
    main()
