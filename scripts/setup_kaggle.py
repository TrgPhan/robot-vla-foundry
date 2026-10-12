"""Install a pinned single-GPU VLA environment inside Kaggle's writable workspace."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(argv):
    print("Run:", argv, flush=True)
    subprocess.run(argv, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="/kaggle/working/robot-vla-env")
    parser.add_argument("--torch-index", choices=["cu118", "cu126", "cu128"], default="cu118")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("This installer targets Kaggle/Linux. Local baseline only needs pip install -e .")
    if not shutil.which("uv"):
        run([sys.executable, "-m", "pip", "install", "uv==0.10.11"])
    env_root = Path(args.env).resolve()
    if not (env_root / "pyvenv.cfg").exists():
        run(["uv", "venv", "--python", "3.12", str(env_root)])
    executable = str(env_root / "bin/python")
    cuda_constraints = env_root / "cuda_constraints.txt"
    cuda_constraints.write_text(
        f"torch==2.7.1+{args.torch_index}\ntorchvision==0.22.1+{args.torch_index}\n", encoding="utf-8"
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            executable,
            "torch==2.7.1",
            "torchvision==0.22.1",
            "--index-url",
            f"https://download.pytorch.org/whl/{args.torch_index}",
        ]
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            executable,
            "-e",
            ".",
            "--constraint",
            "requirements/train_constraints.txt",
            "--constraint",
            str(cuda_constraints),
            "--no-sources",
        ]
    )
    run([executable, "-m", "robot_foundry.cli", "bootstrap", "--repos", "lerobot"])
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            executable,
            "-e",
            "vendor/lerobot[smolvla]",
            "--constraint",
            "requirements/train_constraints.txt",
            "--constraint",
            str(cuda_constraints),
            "--no-sources",
        ]
    )
    run([executable, "-m", "robot_foundry.cli", "doctor"])
    print(f"Training Python: {executable}")


if __name__ == "__main__":
    main()
