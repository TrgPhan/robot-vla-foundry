"""Inspect/compact one failed Kaggle run; preserve usable models and resume checkpoints."""

import argparse
import json
import re
import shutil
from pathlib import Path

from robot_foundry.io import file_hash, fingerprint, read_json
from robot_foundry.packed import build_packed_cache
from robot_foundry.storage import (
    GIB,
    owned_directory,
    recover_export,
    recover_temporary_checkpoint,
    tree_bytes,
)


def inside(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Refusing cache operation outside {root}: {path}")
    return path


def clear_incomplete_exports(out, backend):
    freed = 0
    recover_export(out / "best")
    pending = owned_directory(out / ".best.pending", out)
    if pending.exists() and not (pending / "export_complete.json").exists():
        freed += tree_bytes(pending)
        shutil.rmtree(pending)
    if backend == "smolvla":
        from safetensors import SafetensorError, safe_open

        model = inside(out / "best/model.safetensors", out)
        if model.exists():
            try:
                with safe_open(model, framework="pt", device="cpu") as weights:
                    list(weights.keys())
            except SafetensorError as error:
                if "I/O error" in str(error):
                    raise  # A permission/device failure does not prove the weights are corrupt.
                # Only a proven unreadable export is removed; valid weights are retained.
                print(f"Removing unreadable export {model}: {type(error).__name__}", flush=True)
                freed += model.stat().st_size
                model.unlink()
    return freed


def prune_raw(cfg, prepared, project):
    raw = inside(Path(cfg["raw_dir"]).resolve(), project / "data")
    candidates = {}
    for record in read_json(prepared / "manifest.json")["episodes"]:
        meta = prepared / Path(record["path"]).with_suffix(".json")
        for name, digest in read_json(meta).get("source_files", {}).items():
            path = inside(raw / record["source"] / name, raw)
            if path in candidates and candidates[path] != digest:
                raise ValueError(f"Conflicting raw cache provenance: {path}")
            candidates[path] = digest
    freed = 0
    for path, expected in candidates.items():
        if path.exists():
            if file_hash(path) != expected:
                raise ValueError(f"Raw cache changed; retaining {path}")
            freed += path.stat().st_size
            path.unlink()
    return freed


def repair(project, run, apply=False, remove_raw=False):
    project = Path(project).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run):
        raise ValueError("Invalid run name")
    out = inside(project / "outputs" / run, project / "outputs")
    cfg = read_json(out / "run_config.json")
    # Paths in stored Kaggle recipes are relative to the project, independent of caller cwd.
    prepared = inside(project / cfg["prepared_dir"], project / "data/prepared")
    report = {
        "run": run,
        "free_gib_before": shutil.disk_usage(project).free / GIB,
        "prepared_gib": tree_bytes(prepared) / GIB,
        "raw_gib": tree_bytes(project / cfg["raw_dir"]) / GIB,
        "output_gib": tree_bytes(out) / GIB,
        "resume_checkpoint_exists": (out / "training_state.pt").is_file(),
        "applied": apply,
    }
    print(json.dumps(report, indent=2), flush=True)
    if apply:
        status = recover_temporary_checkpoint(
            out / "training_state.pt", fingerprint(cfg), read_json(prepared / "manifest.json")["fingerprint"]
        )
        report["temporary_checkpoint_status"] = status
        report["resume_checkpoint_exists"] = (out / "training_state.pt").is_file()
        failed = clear_incomplete_exports(out, cfg["backend"])
        duplicated = build_packed_cache(prepared, compact=True)
        raw_freed = 0
        if remove_raw:
            raw_cfg = dict(cfg, raw_dir=str(project / cfg["raw_dir"]))
            raw_freed = prune_raw(raw_cfg, prepared, project)
        report.update(
            incomplete_export_freed_gib=failed / GIB,
            duplicate_npz_freed_gib=duplicated / GIB,
            raw_cache_freed_gib=raw_freed / GIB,
            free_gib_after=shutil.disk_usage(project).free / GIB,
        )
        print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Name from outputs/<run>/run_config.json")
    parser.add_argument("--apply", action="store_true", help="Verify and compact this run's data")
    parser.add_argument(
        "--prune-raw", action="store_true", help="Also remove hash-verified redownloadable raw files"
    )
    args = parser.parse_args()
    repair(Path(__file__).resolve().parents[1], args.run, args.apply, args.prune_raw)


if __name__ == "__main__":
    main()
