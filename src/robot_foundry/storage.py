"""Disk budgets and recoverable checkpoint publication on a single filesystem."""

import os
import shutil
import zipfile
from pathlib import Path

import torch

from .io import read_json, write_json

GIB = 2**30


def require_free(path, required, purpose):
    path = Path(path).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < required:
        raise OSError(
            28,
            f"Insufficient disk for {purpose}: free={free / GIB:.2f} GiB, "
            f"required={required / GIB:.2f} GiB. Run scripts/kaggle_storage.py to compact verified data. "
            "Changing GPU microbatch does not free disk.",
        )
    return free


def tree_bytes(path):
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def archive_directory(source, destination, root):
    source, destination, root = Path(source).resolve(), Path(destination), Path(root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not source.is_relative_to(root) or destination.resolve().is_relative_to(source):
        raise ValueError("Archive source must be inside its root and destination outside the source.")
    files = [p for p in source.rglob("*") if p.is_file()]
    if any(p.is_symlink() for p in files):
        raise ValueError("Refusing symlinks in artifact archive.")
    size = sum(p.stat().st_size for p in files)
    require_free(destination.parent, int(size * 1.01) + 128 * 2**20, "optional ZIP archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(root).as_posix())
        os.replace(temp, destination)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return destination


def save_budget(model, out, export_best=True):
    weights = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    moments = 2 * sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    checkpoint = weights + moments + 32 * 2**20
    export = weights + 32 * 2**20
    # Both writers retain the old artifact while writing its replacement.
    required = checkpoint
    if export_best:
        required = export + max(0, checkpoint - tree_bytes(Path(out) / "best"))
    return required + 256 * 2**20


def recover_temporary_checkpoint(path, config_hash, data_fingerprint):
    """Salvage a complete interrupted rename; remove only an unreadable partial writer file."""
    path = Path(path)
    temp = path.with_suffix(".tmp")
    if not temp.exists():
        return "none"
    if temp.is_symlink():
        raise ValueError("Temporary checkpoint must be a regular owned file.")
    try:
        state = torch.load(temp, map_location="cpu", weights_only=False)
    except OSError:
        raise  # Permission/device errors are not proof that serialization is incomplete.
    except Exception:
        temp.unlink()
        return "removed-incomplete"
    expected = {"config_hash": config_hash, "data_fingerprint": data_fingerprint, "engine": "ddp_amp_v2"}
    if any(state.get(k) != value for k, value in expected.items()):
        raise ValueError(f"Temporary checkpoint belongs to a different recipe: preserving {temp}")
    if not all(k in state for k in ["model", "optimizer", "scheduler", "scaler", "rank_states", "step"]):
        raise ValueError(f"Incomplete checkpoint schema: preserving {temp}")
    step, world_size = state["step"], state["world_size"]
    del state
    if path.exists():
        current = torch.load(path, map_location="cpu", weights_only=False)
        if (
            any(current.get(k) != value for k, value in expected.items())
            or current["world_size"] != world_size
        ):
            raise ValueError("Temporary/current checkpoint provenance differs; preserving both files.")
        older = step <= current["step"]
        del current
        if older:
            temp.unlink()
            return "removed-older"
    os.replace(temp, path)
    return "promoted-complete"


def best_usable(path, backend):
    path = Path(path)
    try:
        for name in ["stats.json", "data_manifest.json"]:
            read_json(path / name)
        if backend == "tiny_bc":
            state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
            if "model" not in state or "spec" not in state:
                return False
        else:
            from safetensors import safe_open

            read_json(path / "config.json")
            with safe_open(path / "model.safetensors", framework="pt", device="cpu") as weights:
                if not weights.keys():
                    return False
        return True
    except Exception:
        return False


def owned_directory(path, parent):
    path, parent = Path(path), Path(parent).resolve()
    if path.is_symlink() or path.resolve().parent != parent:
        raise ValueError(f"Artifact directory must stay inside {parent}: {path}")
    return path


def recover_export(path):
    path = owned_directory(Path(path), Path(path).parent)
    previous = owned_directory(path.with_name(f".{path.name}.previous"), path.parent)
    if previous.exists():
        if not path.exists():
            os.replace(previous, path)
        elif (path / "export_complete.json").exists():
            shutil.rmtree(previous)
        else:
            raise RuntimeError(f"Ambiguous export recovery: preserve {path} and {previous} for inspection.")


def atomic_export(backend, path, dataset, metadata):
    path = owned_directory(Path(path), Path(path).parent)
    stage = owned_directory(path.with_name(f".{path.name}.pending"), path.parent)
    previous = owned_directory(path.with_name(f".{path.name}.previous"), path.parent)
    recover_export(path)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        backend.export(stage, dataset, asset_destination=path / "vlm_assets")
        write_json(stage / "export_complete.json", metadata)
        if path.exists():
            os.replace(path, previous)
        try:
            os.replace(stage, path)
        except BaseException:
            if previous.exists() and not path.exists():
                os.replace(previous, path)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
