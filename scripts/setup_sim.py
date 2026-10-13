"""Create a LIBERO environment without changing the training environment or user's ~/.libero."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(argv):
    print("Run:", argv, flush=True)
    subprocess.run(argv, check=True)


def require_install_space(workspace, torch_index):
    # Conservative installation reserve, including temporary unpacked wheels.
    required = (4 if torch_index == "cpu" else 8) * 2**30
    free = shutil.disk_usage(workspace).free
    if free < required:
        raise OSError(
            28,
            f"Simulation installation needs {required / 2**30:.0f} GiB free; "
            f"available={free / 2**30:.2f} GiB. Preserve outputs and prepared data. "
            "Use --torch-index cpu for the simulator; serve the GPU policy from the training environment.",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=".venv-sim")
    parser.add_argument("--python", default="3.12")
    parser.add_argument("--share-installed-packages", action="store_true")
    parser.add_argument("--torch-index", choices=["cpu", "cu118", "cu126", "cu128"], default="cpu")
    args = parser.parse_args()
    env_root = Path(args.env).resolve()
    workspace = Path.cwd().resolve()
    if not env_root.is_relative_to(workspace) or env_root == workspace:
        raise ValueError("Place the simulation environment inside this project.")
    if not shutil.which("uv"):
        raise RuntimeError("Install uv first: python -m pip install uv")
    require_install_space(workspace, args.torch_index)
    run([sys.executable, "-m", "robot_foundry.cli", "bootstrap", "--repos", "groot"])
    create = ["uv", "venv", "--python", args.python, str(env_root)]
    if args.share_installed_packages:
        create.append("--system-site-packages")
    if not (env_root / "pyvenv.cfg").exists():
        run(create)
    executable = env_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    constraints = env_root / "torch_constraints.txt"
    if not args.share_installed_packages:
        constraints.write_text(
            f"torch==2.7.1+{args.torch_index}\ntorchvision==0.22.1+{args.torch_index}\n", encoding="utf-8"
        )
        run(
            [
                "uv",
                "pip",
                "install",
                "--no-cache",
                "--python",
                str(executable),
                f"torch==2.7.1+{args.torch_index}",
                f"torchvision==0.22.1+{args.torch_index}",
                "numpy==1.26.4",
                "Pillow==11.3.0",
                "--index-url",
                f"https://download.pytorch.org/whl/{args.torch_index}",
            ]
        )
    dependencies = ["uv", "pip", "install", "--no-cache", "--python", str(executable)]
    if not args.share_installed_packages:
        # uv forwards constraints to build subprocesses; avoid a spaced absolute workspace path.
        dependencies += ["--constraint", str(constraints.relative_to(workspace))]
    run([*dependencies, "-r", "requirements/sim.txt"])
    run(
        [
            "uv",
            "pip",
            "install",
            "--no-cache",
            "--python",
            str(executable),
            "--no-deps",
            "-e",
            "vendor/groot/external_dependencies/LIBERO",
            "--config-settings",
            "editable_mode=compat",
        ]
    )
    run(["uv", "pip", "install", "--no-cache", "--python", str(executable), "--no-deps", "-e", "."])
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
    check = (
        "import json, torch, numpy, mujoco, robosuite, matplotlib; "
        "print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, "
        "'numpy': numpy.__version__, 'mujoco': mujoco.__version__}))"
    )
    check_env = dict(os.environ)
    if os.name != "nt":
        check_env.setdefault("MUJOCO_GL", "egl")
        check_env.setdefault("PYOPENGL_PLATFORM", check_env["MUJOCO_GL"])
    result = subprocess.check_output([str(executable), "-c", check], text=True, env=check_env)
    versions = json.loads(result)
    if versions["numpy"] != "1.26.4":
        raise RuntimeError(f"Simulation NumPy mismatch: {versions}")
    if not args.share_installed_packages and versions["torch"] != f"2.7.1+{args.torch_index}":
        raise RuntimeError(f"Simulation PyTorch mismatch: {versions}")
    print("Simulation dependencies:", versions, flush=True)
    print(f"Simulation Python: {executable}")


if __name__ == "__main__":
    main()
