"""Package source and the small tested baseline, excluding data, environments and secrets."""

import json
import zipfile
from pathlib import Path

from robot_foundry.io import file_hash, write_json


def add_file(archive, path, root):
    archive.write(path, path.relative_to(root).as_posix())


def main():
    root = Path.cwd().resolve()
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    source = dist / "robot_foundry_source.zip"
    selected = []
    for folder in ["src", "configs", "scripts", "tests", "requirements", "notebooks", "docs", "reports"]:
        selected.extend(
            p
            for p in (root / folder).rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in [".pyc", ".pt", ".safetensors"]
        )
    selected.extend(root / p for p in ["README.md", "LICENSE", "pyproject.toml", ".gitignore"])
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            if path.is_symlink():
                raise ValueError(f"Unexpected symlink in source package: {path}")
            add_file(archive, path, root)
    baseline = dist / "tiny_bc_checkpoint.zip"
    with zipfile.ZipFile(baseline, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted((root / "outputs/local_tiny_bc/best").rglob("*")):
            if p.is_file():
                archive.write(
                    p, "tiny_bc_best/" + p.relative_to(root / "outputs/local_tiny_bc/best").as_posix()
                )
    artifacts = [
        {"file": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in [source, baseline]
    ]
    write_json(dist / "SHA256.json", artifacts)
    print(json.dumps(artifacts, indent=2))


if __name__ == "__main__":
    main()
