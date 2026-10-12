"""Pinned Hub downloads and strict LeRobot v2/v3 inspection."""

import json
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .io import read_json, write_json


def fetch_file(entry, relative, root):
    root = Path(root)
    return Path(
        hf_hub_download(
            repo_id=entry["repo_id"],
            repo_type="dataset",
            revision=entry["revision"],
            filename=relative,
            local_dir=str(root),
        )
    )


def metadata(entry, root):
    """Fetch small metadata first, not an entire multi-terabyte repository."""
    fetch_file(entry, "meta/info.json", root)
    info = read_json(Path(root) / "meta/info.json")
    version = info["codebase_version"]
    if version.startswith("v2"):
        for name in ["episodes.jsonl", "tasks.jsonl"]:
            fetch_file(entry, "meta/" + name, root)
        episodes = [
            json.loads(line) for line in (Path(root) / "meta/episodes.jsonl").read_text().splitlines() if line
        ]
        tasks = {
            r["task_index"]: r["task"]
            for r in [
                json.loads(line)
                for line in (Path(root) / "meta/tasks.jsonl").read_text().splitlines()
                if line
            ]
        }
    elif version.startswith("v3"):
        # Enumerate pinned metadata files: v3 episode tables can span many chunks.
        for name in entry["files"]:
            if (
                name.startswith("meta/episodes/")
                and name.endswith(".parquet")
                or name == "meta/tasks.parquet"
            ):
                fetch_file(entry, name, root)
        episodes = []
        for p in sorted((Path(root) / "meta/episodes").rglob("*.parquet")):
            episodes.extend(pq.read_table(p).to_pylist())
        tasks = {}
        for row in pq.read_table(Path(root) / "meta/tasks.parquet").to_pylist():
            # pandas serializes its task text index as __index_level_0__ or task.
            text_key = "task" if "task" in row else "__index_level_0__"
            tasks[int(row["task_index"])] = row[text_key]
    else:
        raise ValueError(f"Unsupported LeRobot version: {version}")
    if len(episodes) != info["total_episodes"]:
        raise ValueError("Episode metadata incomplete; refusing a partial split.")
    return info, episodes, tasks


def signature(info):
    return {
        "robot_type": info["robot_type"],
        "fps": info["fps"],
        "state": info["features"]["observation.state"],
        "action": info["features"]["action"],
        "cameras": sorted(k for k, v in info["features"].items() if v["dtype"] == "video"),
    }


def compatible(a, b):
    """Require an explicitly shared action convention in addition to matching shapes."""
    sa, sb = signature(a), signature(b)
    if sa["robot_type"] != sb["robot_type"] or sa["fps"] != sb["fps"] or sa["cameras"] != sb["cameras"]:
        raise ValueError("Robot, FPS, or camera mismatch: use separate embodiment recipes.")
    for key in ["state", "action"]:
        if any(sa[key].get(k) != sb[key].get(k) for k in ["shape", "names", "dtype"]):
            raise ValueError(f"{key} ordering/type mismatch; equal vector sizes are insufficient.")


def episode_paths(info, episode, cameras):
    eid = int(episode["episode_index"])
    if info["codebase_version"].startswith("v2"):
        kwargs = {"episode_index": eid, "episode_chunk": eid // info["chunks_size"]}
        data = info["data_path"].format(**kwargs)
        videos = {key: (info["video_path"].format(**kwargs, video_key=key), 0.0) for key in cameras}
    else:
        data = info["data_path"].format(
            chunk_index=int(episode["data/chunk_index"]), file_index=int(episode["data/file_index"])
        )
        videos = {
            key: (
                info["video_path"].format(
                    video_key=key,
                    chunk_index=int(episode[f"videos/{key}/chunk_index"]),
                    file_index=int(episode[f"videos/{key}/file_index"]),
                ),
                float(episode[f"videos/{key}/from_timestamp"]),
            )
            for key in cameras
        }
    return data, videos


def load_episode(entry, root, info, episode, cameras):
    data_path, video_paths = episode_paths(info, episode, cameras)
    table = pq.read_table(
        fetch_file(entry, data_path, root),
        columns=["observation.state", "action", "episode_index", "frame_index", "timestamp", "task_index"],
    )
    table = table.filter(pc.equal(table["episode_index"], episode["episode_index"]))
    rows = sorted(table.to_pylist(), key=lambda r: r["frame_index"])
    if len(rows) != int(episode["length"]) or not rows:
        raise ValueError("Missing rows or invalid episode length.")
    if [int(r["frame_index"]) for r in rows] != list(range(len(rows))):
        raise ValueError("Non-contiguous frame indices.")
    states = np.asarray([r["observation.state"] for r in rows], dtype=np.float32)
    actions = np.asarray([r["action"] for r in rows], dtype=np.float32)
    timestamps = np.asarray([r["timestamp"] for r in rows], dtype=np.float64)
    task_indices = np.asarray([r["task_index"] for r in rows], dtype=np.int64)
    if not np.isfinite(states).all() or not np.isfinite(actions).all() or not np.isfinite(timestamps).all():
        raise ValueError("Non-finite state/action/timestamp.")
    if states.shape[1:] != tuple(info["features"]["observation.state"]["shape"]) or actions.shape[
        1:
    ] != tuple(info["features"]["action"]["shape"]):
        raise ValueError("State/action dimensions disagree with metadata.")
    if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 1 / info["fps"], atol=1e-3):
        raise ValueError("Irregular timestamps; resample explicitly before training.")
    for path, _ in video_paths.values():
        fetch_file(entry, path, root)
    return states, actions, timestamps, task_indices, video_paths


def split_episodes(episodes, seed, per_task=None):
    """Whole-episode, task-stratified split. At least one validation and test episode per task."""
    groups = {}
    for ep in episodes:
        tasks = ep.get("tasks")
        if not tasks or len(tasks) != 1:
            raise ValueError("This recipe requires one language task per episode.")
        groups.setdefault(tasks[0], []).append(ep)
    rng = np.random.default_rng(seed)
    result = {k: [] for k in ["train", "val", "test"]}
    for task in sorted(groups):
        eps = sorted(groups[task], key=lambda e: e["episode_index"])
        rng.shuffle(eps)
        if per_task is not None:
            eps = eps[:per_task]
        if len(eps) < 3:
            raise ValueError(f"Task {task!r} has <3 episodes; cannot make an honest 3-way split.")
        n_hold = max(1, round(len(eps) * 0.1))
        result["test"].extend(eps[:n_hold])
        result["val"].extend(eps[n_hold : 2 * n_hold])
        result["train"].extend(eps[2 * n_hold :])
    return result


def audit_registry(registry_path, output):
    from huggingface_hub import HfApi

    registry = read_json(registry_path)
    report = []
    for key, entry in registry.items():
        kind = entry.get("kind", "dataset")
        meta = HfApi().repo_info(entry["repo_id"], repo_type=kind, revision=entry["revision"])
        report.append(
            {
                "key": key,
                "repo_id": entry["repo_id"],
                "resolved_sha": meta.sha,
                "gated": meta.gated,
                "role": entry["role"],
                "license": entry.get("license"),
            }
        )
    write_json(output, report)
    return report
