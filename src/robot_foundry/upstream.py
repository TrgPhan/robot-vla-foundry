"""Pinned upstream checkout helpers without training or simulator imports."""

import os
import subprocess
from pathlib import Path

from .io import read_json


def check_checkout(key, path, lock_path="configs/upstream_lock.json"):
    lock = read_json(lock_path)[key]
    actual = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if actual != lock["sha"]:
        raise ValueError(f"{key} revision mismatch: expected {lock['sha']}, got {actual}")
    return lock


def bootstrap(keys, vendor="vendor"):
    locks = read_json("configs/upstream_lock.json")
    for key in keys:
        entry = locks[key]
        root = Path(vendor) / key
        if (root / ".git").exists():
            check_checkout(key, root)
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        if not (root / ".git").exists():
            if root.exists() and any(root.iterdir()):
                raise FileExistsError(f"Nonempty destination: {root}")
            root.mkdir(parents=True, exist_ok=True)
            for args in [
                ["init"],
                ["remote", "add", "origin", f"https://github.com/{entry['repo']}.git"],
                ["fetch", "--depth", "1", "origin", entry["sha"]],
                ["checkout", "--detach", "FETCH_HEAD"],
            ]:
                subprocess.run(
                    ["git", "-c", "core.longpaths=true", "-C", str(root), *args], env=env, check=True
                )
        # Do not install large CUDA/Isaac dependencies into the user's existing Python environment.
        if key == "groot":
            # Only LIBERO is used by these recipes; other simulators have large, unrelated asset trees.
            subprocess.run(
                [
                    "git",
                    "-c",
                    "core.longpaths=true",
                    "-C",
                    str(root),
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "external_dependencies/LIBERO",
                ],
                env=env,
                check=True,
            )
        check_checkout(key, root)
