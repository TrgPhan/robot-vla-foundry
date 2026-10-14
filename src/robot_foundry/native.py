"""Export honest train-only native GR00T datasets and launch its reference implementation."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .datasets import episode_paths
from .hardware import require_gpu
from .io import read_json, write_json
from .upstream import bootstrap as bootstrap, check_checkout as check_checkout


def materialize(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def transcode_h264(source, dest):
    import av

    dest.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(source)) as src, av.open(str(dest), mode="w") as dst:
        in_stream = src.streams.video[0]
        out_stream = dst.add_stream("libx264", rate=in_stream.average_rate)
        out_stream.width, out_stream.height = in_stream.width, in_stream.height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "18", "preset": "fast"}
        for frame in src.decode(in_stream):
            frame.pts = None
            for packet in out_stream.encode(frame):
                dst.mux(packet)
        for packet in out_stream.encode():
            dst.mux(packet)


def export_native(prepared_dir, output_dir, groot_repo, h264=True):
    check_checkout("groot", groot_repo)
    manifest = read_json(Path(prepared_dir) / "manifest.json")
    if manifest["action_convention"] != "libero_osc_delta_xyz_axis_angle_gripper_open01":
        raise ValueError("This native export is specifically for the validated LIBERO Panda convention.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a new export directory: {output_dir}")
    exported = []
    for source in sorted({r["source"] for r in manifest["episodes"]}):
        raw_root = Path(manifest["config"]["raw_dir"]) / source
        info = read_json(raw_root / "meta/info.json")
        if not info["codebase_version"].startswith("v2"):
            raise ValueError(
                "Reference GR00T needs LeRobot v2. Use its official v3 converter in a separate env."
            )
        episode_map = {
            r["episode_index"]: r
            for r in [json.loads(s) for s in (raw_root / "meta/episodes.jsonl").read_text().splitlines()]
        }
        train_rows = [r for r in manifest["episodes"] if r["source"] == source and r["split"] == "train"]
        numeric = {"observation.state": [], "action": []}
        for row in train_rows:
            with np.load(Path(prepared_dir) / row["path"], allow_pickle=False) as arr:
                numeric["observation.state"].append(arr["full_states"])
                numeric["action"].append(arr["full_actions"])
        stats = {}
        for key, arrays in numeric.items():
            a = np.concatenate(arrays).astype(np.float64)
            stats[key] = {
                "mean": a.mean(0).tolist(),
                "std": a.std(0).clip(1e-4).tolist(),
                "min": a.min(0).tolist(),
                "max": a.max(0).tolist(),
                "q01": np.quantile(a, 0.01, axis=0).tolist(),
                "q99": np.quantile(a, 0.99, axis=0).tolist(),
            }
        for split in ["train", "val", "test"]:
            root = output_dir / source / split
            rows = [r for r in manifest["episodes"] if r["source"] == source and r["split"] == split]
            ep_lines = []
            frame_offset = 0
            for new_id, row in enumerate(rows):
                old = episode_map[row["episode_index"]]
                data, videos = episode_paths(info, old, manifest["signature"]["cameras"])
                table = pq.read_table(raw_root / data)
                table = table.set_column(
                    table.schema.get_field_index("episode_index"),
                    "episode_index",
                    pa.array([new_id] * len(table), type=pa.int64()),
                )
                table = table.set_column(
                    table.schema.get_field_index("index"),
                    "index",
                    pa.array(range(frame_offset, frame_offset + len(table)), type=pa.int64()),
                )
                kwargs = {"episode_index": new_id, "episode_chunk": new_id // info["chunks_size"]}
                dest = root / info["data_path"].format(**kwargs)
                dest.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, dest)
                for key, (video, _) in videos.items():
                    dst = root / info["video_path"].format(**kwargs, video_key=key)
                    if h264:
                        transcode_h264(raw_root / video, dst)
                    else:
                        materialize(raw_root / video, dst)
                ep_lines.append({**old, "episode_index": new_id})
                frame_offset += len(table)
            new_info = json.loads(json.dumps(info))
            new_info.update(
                total_episodes=len(rows),
                total_frames=frame_offset,
                total_videos=len(rows) * len(manifest["signature"]["cameras"]),
                total_chunks=(len(rows) + info["chunks_size"] - 1) // info["chunks_size"],
                splits={"train": f"0:{len(rows)}"},
            )
            if h264:
                for v in new_info["features"].values():
                    if v["dtype"] == "video":
                        v["info"]["video.codec"] = "h264"
            write_json(root / "meta/info.json", new_info)
            write_json(root / "meta/stats.json", stats)
            (root / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ep_lines))
            shutil.copy2(raw_root / "meta/tasks.jsonl", root / "meta/tasks.jsonl")
            shutil.copy2(Path(groot_repo) / "examples/LIBERO/modality.json", root / "meta/modality.json")
            write_json(
                root / "meta/provenance.json",
                {
                    "split": split,
                    "source": source,
                    "training_stats_only": True,
                    "manifest_fingerprint": manifest["fingerprint"],
                    "episodes": rows,
                },
            )
            if split == "train":
                exported.append(str(root.resolve()))
    write_json(
        output_dir / "export_manifest.json",
        {"train_paths": exported, "source_manifest": manifest["fingerprint"], "h264": h264},
    )
    return exported


def validate_g1_dataset(root):
    """Validate the actual 64+7+7 SONIC action layout; never fabricate tokens from arm angles."""
    root = Path(root)
    info = read_json(root / "meta/info.json")
    modality = read_json(root / "meta/modality.json")
    if not info["codebase_version"].startswith("v2") or abs(info["fps"] - 50) > 1e-3:
        raise ValueError("SONIC recipe expects LeRobot v2 at 50 Hz; use NVIDIA's collection/export stack.")
    expected = {"motion_token": 64, "left_hand_joints": 7, "right_hand_joints": 7}
    numeric_columns = set()
    occupied = {}

    def validate_slice(kind, key, dim, default_column):
        field = modality.get(kind, {}).get(key)
        if not isinstance(field, dict):
            raise ValueError(f"Missing/invalid SONIC {kind} field {key}: expected {dim} values.")
        start, end = field.get("start"), field.get("end")
        if type(start) is not int or type(end) is not int or start < 0 or end - start != dim:
            raise ValueError(f"Invalid {kind} slice {key}: expected {dim} values at a nonnegative offset.")
        column = field.get("original_key", default_column)
        feature = info["features"].get(column)
        if feature is None or len(feature["shape"]) != 1 or feature["shape"][0] < end:
            raise ValueError(f"{kind} mapping refers to an absent or short feature: {column}")
        indexes = set(range(start, end))
        used = occupied.setdefault((kind, column), set())
        if used & indexes:
            raise ValueError(f"Overlapping G1 {kind} slices in {column}: {key}")
        used.update(indexes)
        numeric_columns.add(column)

    for key, dim in expected.items():
        validate_slice("action", key, dim, "action")
    state_dims = {
        "left_leg": 6,
        "right_leg": 6,
        "waist": 3,
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 7,
        "right_hand": 7,
        "projected_gravity": 3,
    }
    for key, dim in state_dims.items():
        validate_slice("state", key, dim, "observation.state")
    if "ego_view" not in modality.get("video", {}):
        raise ValueError("Missing G1 ego_view camera.")
    # Identity is mandatory because the token codebook must match the deployed controller.
    provenance = read_json(root / "meta/sonic_provenance.json")
    if provenance.get("controller_variant") != "sonic_v1_1" or not provenance.get("controller_sha256"):
        raise ValueError("Record the exact SONIC v1.1 controller digest in meta/sonic_provenance.json.")
    digest = provenance["controller_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError("controller_sha256 must be a SHA-256 hex digest of the collected-data controller.")
    frames = 0
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=sorted(numeric_columns))
        for key in numeric_columns:
            array = np.asarray(table[key].to_pylist())
            if array.shape != (len(table), info["features"][key]["shape"][0]) or not np.isfinite(array).all():
                raise ValueError(f"Non-finite or malformed G1 labels in {path}: {key}")
        frames += len(table)
    if not frames or frames != info["total_frames"]:
        raise ValueError("G1 parquet frames are missing or disagree with metadata.")
    return {"action_dimensions": 78, "controller": provenance, "fps": info["fps"], "checked_frames": frames}


def groot_command(cfg, repo, dataset_paths, model_path):
    # Paths use the reference launcher's os.pathsep grammar, not invented repeated flags.
    cmd = [
        sys.executable,
        "gr00t/experiment/launch_finetune.py",
        "--base-model-path",
        str(model_path),
        "--dataset-path",
        os.pathsep.join(str(Path(p).resolve()) for p in dataset_paths),
        "--embodiment-tag",
        cfg["embodiment_tag"],
        "--output-dir",
        str(Path(cfg["output_dir"]).resolve()),
        "--num-gpus",
        str(cfg["num_gpus"]),
        "--global-batch-size",
        str(cfg["global_batch_size"]),
        "--gradient-accumulation-steps",
        str(cfg["gradient_accumulation_steps"]),
        "--max-steps",
        str(cfg["steps"]),
        "--save-steps",
        str(cfg["save_steps"]),
        "--save-total-limit",
        "3",
        "--learning-rate",
        str(cfg["lr"]),
        "--dataloader-num-workers",
        str(cfg.get("num_workers", 2)),
        "--num-shards-per-epoch",
        str(cfg.get("num_shards_per_epoch", 64)),
        "--state-dropout-prob",
        "0.2",
        "--ds-weights-alpha",
        "0.0",
    ]
    if cfg.get("resume", False):
        cmd.append("--resume-from-checkpoint")
    return cmd


def launch_groot(config_path, dry_run=False):
    cfg = read_json(config_path)
    repo = Path(cfg["groot_repo"]).resolve()
    check_checkout("groot", repo)
    if "export_manifest" in cfg:
        paths = read_json(cfg["export_manifest"])["train_paths"]
    else:
        paths = cfg["dataset_paths"]
    if cfg["embodiment_tag"] == "UNITREE_G1_SONIC":
        for p in paths:
            validate_g1_dataset(p)
    for p in paths:
        info = read_json(Path(p) / "meta/info.json")
        if not info["codebase_version"].startswith("v2"):
            raise ValueError("GR00T reference loader needs v2, not a raw v3 dataset.")
        for name in ["stats.json", "modality.json", "episodes.jsonl", "tasks.jsonl"]:
            if not (Path(p) / "meta" / name).exists():
                raise FileNotFoundError(Path(p) / "meta" / name)
    registry = read_json("configs/datasets.json")
    model = registry["groot"]
    model_path = Path(cfg.get("model_cache", "data/models/groot")).resolve()
    cmd = groot_command(cfg, repo, paths, model_path)
    plan = {
        "cwd": str(repo),
        "argv": cmd,
        "model_repo": model["repo_id"],
        "model_revision": model["revision"],
        "effective_batch": cfg["global_batch_size"] * cfg["gradient_accumulation_steps"],
    }
    print(json.dumps(plan, indent=2))
    if dry_run:
        return plan
    require_gpu("groot", cfg["num_gpus"])
    if cfg["global_batch_size"] % cfg["num_gpus"]:
        raise ValueError("Global batch size must be divisible by num_gpus.")
    from huggingface_hub import hf_hub_download, snapshot_download

    # Fail on gated backbone access before downloading the larger base model.
    hf_hub_download("nvidia/Cosmos-Reason2-2B", "config.json")
    snapshot_download(model["repo_id"], revision=model["revision"], local_dir=str(model_path))
    write_json(Path(cfg["output_dir"]) / "launch_plan.json", plan)
    subprocess.run(cmd, cwd=repo, check=True)
    return plan
