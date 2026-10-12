"""Create a LIBERO environment without changing the training environment or user's ~/.libero."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(argv):
    print("Run:", argv, flush=True)
    subprocess.run(argv, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".venv-sim")
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--share-installed-packages", action="store_true")
    parser.add_argument("--torch-index", choices=["cu118", "cu126", "cu128"], default="cu118")
    args = parser.parse_args()
    env_root = Path(args.env).resolve()
    workspace = Path.cwd().resolve()
    if not env_root.is_relative_to(workspace) or env_root == workspace:
        raise ValueError("Place the simulation environment inside this project.")
    if not shutil.which("uv"):
        raise RuntimeError("Install uv first: python -m pip install uv")
    run([sys.executable, "-m", "robot_foundry.cli", "bootstrap", "--repos", "groot"])
    create = ["uv", "venv", "--python", args.python, str(env_root)]
    if args.share_installed_packages:
        create.append("--system-site-packages")
    if not (env_root / "pyvenv.cfg").exists():
        run(create)
    executable = env_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not args.share_installed_packages:
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(executable),
                "torch==2.7.1",
                "torchvision==0.22.1",
                "--index-url",
                f"https://download.pytorch.org/whl/{args.torch_index}",
            ]
        )
    run(["uv", "pip", "install", "--python", str(executable), "-r", "requirements/sim.txt"])
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(executable),
            "--no-deps",
            "-e",
            "vendor/groot/external_dependencies/LIBERO",
            "--config-settings",
            "editable_mode=compat",
        ]
    )
    run(["uv", "pip", "install", "--python", str(executable), "--no-deps", "-e", "."])
    if os.name == "nt":
        site = env_root / "Lib/site-packages"
        # robosuite 1.4 hardcodes a DLL location and selects EGL before loading macros_private.
        # Apply narrowly scoped fixes to this environment's dependency, not the upstream checkout.
        shutil.copy2(site / "mujoco/mujoco.dll", site / "robosuite/utils/mujoco.dll")
        macros = site / "robosuite/macros.py"
        text = macros.read_text(encoding="utf-8")
        expected = "MUJOCO_GPU_RENDERING = True"
        if expected not in text and "MUJOCO_GPU_RENDERING = False" not in text:
            raise ValueError("Unexpected robosuite macros; review compatibility patch.")
        macros.write_text(text.replace(expected, "MUJOCO_GPU_RENDERING = False"), encoding="utf-8")
    print(f"Simulation Python: {executable}")


if __name__ == "__main__":
    main()
