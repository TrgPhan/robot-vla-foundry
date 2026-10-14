"""Collect paired RGB/proprioception/latent demonstrations from actual G1 SONIC dynamics."""

import json
import os
from pathlib import Path

import numpy as np

from .fullbody import SonicController, load_motion
from .g1_env import G1Environment, scenario_motion
from .g1_schema import ACTION_CONVENTION, CAMERA, ROBOT, STATE_LAYOUT, controller_identity, live_state
from .io import file_hash, fingerprint, read_json, write_json
from .packed import build_episode, cache_path, compact_episode, validate_cache
from .prepare import action_chunk
from .sonic_assets import prepare_sonic_assets


class CameraVideo:
    def __init__(self, path, size, fps=50):
        import av

        self.av = av
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temp = self.path.with_suffix(".tmp.mp4")
        self.container = av.open(str(self.temp), "w")
        self.stream = self.container.add_stream("libx264", rate=fps)
        self.stream.width = self.stream.height = size
        self.stream.pix_fmt = "yuv420p"
        self.stream.options = {"crf": "20", "preset": "fast"}

    def write(self, pixels):
        frame = self.av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        try:
            for packet in self.stream.encode():
                self.container.mux(packet)
        finally:
            self.container.close()
        os.replace(self.temp, self.path)


def collect_episode(scene, source_motion, models, seed, cfg, video_path):
    motion = scenario_motion(source_motion, seed)
    env = G1Environment(scene, motion, seed, cfg["image_size"])
    controller = SonicController(models)
    video = CameraVideo(video_path, cfg["image_size"])
    states, tokens, images, telemetry = [], [], [], []
    frames = min(cfg["max_frames"], len(motion["joint_pos"]))
    try:
        for _ in range(cfg["warmup_frames"]):
            action, _ = controller.predict(*env.proprio(), motion, 0, playing=False)
            if env.step(action)["fall"]:
                raise RuntimeError("Teacher fell during initialization; no demonstration accepted")
        for frame in range(frames):
            # Capture PRE-action state/image; never use a future pose or post-action frame as input.
            states.append(live_state(env.state()))
            pixels = env.image()
            video.write(pixels)
            if frame % cfg["stride"] == 0:
                images.append(pixels.transpose(2, 0, 1).copy())
            action, token = controller.predict(*env.proprio(), motion, frame)
            tokens.append(token.copy())
            metric = env.step(action)
            telemetry.append(metric)
            if metric["fall"]:
                raise RuntimeError(
                    f"Teacher fell at frame {frame}; failed rollout retained in collection log"
                )
    finally:
        try:
            video.close()
        finally:
            env.close()
    states, tokens = np.stack(states), np.stack(tokens).astype(np.float32)
    ids = np.arange(0, frames, cfg["stride"])
    chunks, valid = zip(*(action_chunk(tokens, int(i), cfg["horizon"]) for i in ids))
    arrays = {
        "images": np.stack(images)[:, None],
        "states": states[ids],
        "actions": np.stack(chunks),
        "valid": np.stack(valid),
        "full_states": states,
        "full_actions": tokens,
    }
    return arrays, {
        "frames": frames,
        "max_tilt_deg": max(x["tilt_deg"] for x in telemetry),
        "min_height_m": min(x["height_m"] for x in telemetry),
        "fall": False,
    }


def collect_dataset(config_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cfg = read_json(config_path)
    if cfg.get("collection_version") != 1:
        raise ValueError("Use the versioned configs/g1_data.json recipe for this paired-data exporter")
    if cfg["robot"] != ROBOT or cfg["action_convention"] != ACTION_CONVENTION:
        raise ValueError("Collection config must explicitly select G1 SONIC motion29")
    if any(cfg[x] < 1 for x in ["stride", "horizon", "max_frames"]) or cfg["image_size"] < 32:
        raise ValueError("Invalid collection dimensions")
    if set(cfg["split_rollouts_per_task"]) != {"train", "val", "test"} or any(
        x < 1 for x in cfg["split_rollouts_per_task"].values()
    ):
        raise ValueError("Collect at least one whole episode/task in every split")
    root, raw = Path(cfg["prepared_dir"]), Path(cfg["raw_dir"])
    root.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    state_file = root / "collection_state.json"
    state = (
        read_json(state_file)
        if state_file.exists()
        else {
            "config_fingerprint": fingerprint(cfg),
            "records": [],
            "failed": [],
        }
    )
    if state["config_fingerprint"] != fingerprint(cfg):
        raise ValueError("Collection config changed; select a new raw/prepared directory")
    complete = {r["episode_index"]: r for r in state["records"]}
    identity, provenances, records = None, {}, []
    episode = 0
    for task_id, task in enumerate(cfg["tasks"]):
        scene, motion_path, models, provenance = prepare_sonic_assets(
            cfg["repo"], cfg["assets"], task["motion"]
        )
        current_identity = controller_identity(provenance)
        if identity is not None and identity != current_identity:
            raise ValueError("Do not mix SONIC controller/codebook variants in one dataset")
        identity = current_identity
        provenances[task["motion"]] = provenance
        motion = load_motion(motion_path)
        trial = 0
        for split in ["train", "val", "test"]:
            for _ in range(cfg["split_rollouts_per_task"][split]):
                seed = cfg["seed"] + task_id * 10000 + trial
                trial += 1
                if episode in complete:
                    record = complete[episode]
                    validate_cache(root, record)
                    for kind, filename in [
                        ("parquet", raw / f"data/chunk-000/episode_{episode:06d}.parquet"),
                        ("video", raw / f"videos/chunk-000/{CAMERA}/episode_{episode:06d}.mp4"),
                    ]:
                        if not filename.is_file() or file_hash(filename) != record["raw_sha256"][kind]:
                            raise ValueError(
                                f"Raw G1 artifact missing/changed; restore it or use a new dataset: {filename}"
                            )
                else:
                    path = root / f"episodes/g1_{episode:06d}.npz"
                    video = raw / f"videos/chunk-000/{CAMERA}/episode_{episode:06d}.mp4"
                    print(
                        f"COLLECT (not train): task={task_id} episode={episode} split={split} seed={seed}",
                        flush=True,
                    )
                    try:
                        arrays, metric = collect_episode(scene, motion, models, seed, cfg, video)
                    except Exception as error:
                        state["failed"].append(
                            {"episode": episode, "task": task["motion"], "seed": seed, "error": str(error)}
                        )
                        write_json(state_file, state)
                        raise
                    arrays["prompts"] = np.full(len(arrays["states"]), task["prompt"])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temp, **arrays)
                    os.replace(temp, path)
                    frames = len(arrays["full_states"])
                    table = pa.table(
                        {
                            "observation.state": pa.array(
                                arrays["full_states"].tolist(), type=pa.list_(pa.float32(), 70)
                            ),
                            "action": pa.array(
                                arrays["full_actions"].tolist(), type=pa.list_(pa.float32(), 64)
                            ),
                            "timestamp": np.arange(frames, dtype=np.float32) / 50,
                            "episode_index": np.full(frames, episode, np.int64),
                            "frame_index": np.arange(frames, dtype=np.int64),
                            "task_index": np.full(frames, task_id, np.int64),
                            "index": np.arange(frames, dtype=np.int64) + sum(r["length"] for r in records),
                        }
                    )
                    parquet = raw / f"data/chunk-000/episode_{episode:06d}.parquet"
                    parquet.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table, parquet)
                    record = {
                        "source": task["motion"],
                        "episode_index": episode,
                        "task_id": task_id,
                        "tasks": [task["prompt"]],
                        "split": split,
                        "seed": seed,
                        "samples": len(arrays["states"]),
                        "length": frames,
                        "path": path.relative_to(root).as_posix(),
                        "sha256": file_hash(path),
                        "raw_sha256": {"parquet": file_hash(parquet), "video": file_hash(video)},
                        "teacher_rollout": metric,
                    }
                    build_episode(root, record)
                    compact_episode(root, record)
                    state["records"].append(record)
                    write_json(state_file, state)
                records.append(record)
                episode += 1
    train_states, train_actions = [], []
    for record in records:
        if record["split"] == "train":
            packed = cache_path(root, record)
            train_states.append(np.load(packed / "full_states.npy", allow_pickle=False))
            train_actions.append(np.load(packed / "full_actions.npy", allow_pickle=False))
    stats = {}
    for key, values in [("observation.state", train_states), ("action", train_actions)]:
        x = np.concatenate(values).astype(np.float64)
        stats[key] = {
            "mean": x.mean(0).tolist(),
            "std": x.std(0).clip(1e-4).tolist(),
            "min": x.min(0).tolist(),
            "max": x.max(0).tolist(),
            "count": [len(x)],
        }
    write_json(root / "stats.json", stats)
    manifest = {
        "config": cfg,
        "episodes": records,
        "action_convention": ACTION_CONVENTION,
        "signature": {
            "robot_type": ROBOT,
            "fps": 50,
            "cameras": [CAMERA],
            "state": {"shape": [70], "dtype": "float32"},
            "action": {"shape": [64], "dtype": "float32"},
        },
        "state_layout": STATE_LAYOUT,
        "controller_identity": identity,
        "source_provenance": provenances,
        "split_counts": {s: sum(r["split"] == s for r in records) for s in ["train", "val", "test"]},
        "normalization": "train episodes only; full 50 Hz frames; no padded/repeated labels",
        "scope": "simulation-generated language-conditioned locomotion/gesture; no object manipulation",
        "benchmark_scope": "held-out rollout seeds/camera/initial pose on seen source motion primitives",
        "hands_actuated": False,
        "teacher_rejected_rollouts": state["failed"],
        "statistics_sha256": file_hash(root / "stats.json"),
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(root / "manifest.json", manifest)
    info = {
        "codebase_version": "v2.1",
        "robot_type": ROBOT,
        "fps": 50,
        "total_episodes": len(records),
        "total_frames": sum(r["length"] for r in records),
        "total_tasks": len(cfg["tasks"]),
        "total_videos": len(records),
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [70]},
            "action": {"dtype": "float32", "shape": [64]},
            CAMERA: {"dtype": "video", "shape": [cfg["image_size"], cfg["image_size"], 3]},
            **{
                k: {"dtype": dtype, "shape": [1]}
                for k, dtype in [
                    ("timestamp", "float32"),
                    ("episode_index", "int64"),
                    ("frame_index", "int64"),
                    ("task_index", "int64"),
                    ("index", "int64"),
                ]
            },
        },
    }
    write_json(raw / "meta/info.json", info)
    write_json(raw / "meta/stats.json", stats)
    write_json(raw / "meta/g1_motion_provenance.json", manifest)
    for filename, entries in [
        (
            "episodes",
            [
                {
                    "episode_index": r["episode_index"],
                    "length": r["length"],
                    "tasks": r["tasks"],
                    "split": r["split"],
                }
                for r in records
            ],
        ),
        ("tasks", [{"task_index": i, "task": t["prompt"]} for i, t in enumerate(cfg["tasks"])]),
    ]:
        (raw / f"meta/{filename}.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in entries), encoding="utf-8"
        )
    print(f"COLLECTION COMPLETE: {root / 'manifest.json'} splits={manifest['split_counts']}", flush=True)
    return manifest
