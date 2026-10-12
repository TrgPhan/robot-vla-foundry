"""Prepare timestamp-aligned RGB and padded action chunks with train-only statistics."""

import os
from pathlib import Path

import numpy as np

from .datasets import compatible, episode_paths, load_episode, metadata, signature, split_episodes
from .io import file_hash, fingerprint, read_json, write_json
from .video import decode_frames


def action_chunk(actions, start, horizon):
    if not 0 <= start < len(actions) or horizon < 1:
        raise ValueError("Action chunks need a valid start within a nonempty episode and a positive horizon.")
    indices = np.arange(start, start + horizon)
    valid = indices < len(actions)
    return actions[np.minimum(indices, len(actions) - 1)], valid


def prepare(config_path):
    cfg = read_json(config_path)
    registry = read_json(cfg.get("registry", "configs/datasets.json"))
    out = Path(cfg["prepared_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if cfg["horizon"] < 1 or cfg["stride"] < 1 or cfg["image_size"] < 32:
        raise ValueError("Invalid horizon/stride/image_size.")
    reference = None
    records = []
    sum_state = sum_state2 = sum_action = sum_action2 = None
    count = 0
    for source_i, key in enumerate(cfg["datasets"]):
        entry = registry[key]
        if entry.get("action_convention") != cfg["action_convention"]:
            raise ValueError(f"{key}: action convention must match recipe, including units and gripper sign.")
        root = Path(cfg["raw_dir"]) / key
        info, episodes, tasks = metadata(entry, root)
        if reference is None:
            reference = info
        else:
            compatible(reference, info)
        cameras = sorted(k for k, v in info["features"].items() if v["dtype"] == "video")
        splits = split_episodes(episodes, cfg["seed"] + source_i, cfg.get("episodes_per_task"))
        for split, eps in splits.items():
            for ep in eps:
                eid = int(ep["episode_index"])
                identity = {
                    "revision": entry["revision"],
                    "source": key,
                    "episode": eid,
                    "size": cfg["image_size"],
                    "stride": cfg["stride"],
                    "horizon": cfg["horizon"],
                    "preparation_version": 1,
                }
                cache_key = fingerprint(identity)
                relative = f"episodes/{key}_{eid:06d}_{cache_key[:12]}.npz"
                path = out / relative
                meta_path = path.with_suffix(".json")
                if not path.exists() or not meta_path.exists():
                    states, actions, times, task_ids, videos = load_episode(entry, root, info, ep, cameras)
                    ids = np.arange(0, len(states), cfg["stride"])
                    images = np.stack(
                        [
                            decode_frames(
                                root / videos[c][0], times[ids] + videos[c][1], cfg["image_size"], info["fps"]
                            )
                            for c in cameras
                        ],
                        axis=1,
                    )
                    chunks, masks = zip(*(action_chunk(actions, int(i), cfg["horizon"]) for i in ids))
                    prompts = np.asarray([tasks[int(i)] for i in task_ids[ids]])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp = path.with_suffix(".tmp.npz")
                    np.savez_compressed(
                        temp,
                        images=images,
                        states=states[ids],
                        actions=np.stack(chunks),
                        valid=np.stack(masks),
                        prompts=prompts,
                        full_states=states,
                        full_actions=actions,
                    )
                    os.replace(temp, path)
                    write_json(
                        meta_path,
                        {
                            **identity,
                            "source_files": {
                                p: file_hash(root / p)
                                for p in [
                                    episode_paths(info, ep, cameras)[0],
                                    *[v[0] for v in videos.values()],
                                ]
                            },
                        },
                    )
                with np.load(path, allow_pickle=False) as arrays:
                    n = len(arrays["states"])
                    if split == "train":
                        states = arrays["full_states"].astype(np.float64)
                        actions = arrays["full_actions"].astype(np.float64)
                        if count == 0:
                            sum_state, sum_state2 = states.sum(0), np.square(states).sum(0)
                            sum_action, sum_action2 = actions.sum(0), np.square(actions).sum(0)
                        else:
                            sum_state += states.sum(0)
                            sum_state2 += np.square(states).sum(0)
                            sum_action += actions.sum(0)
                            sum_action2 += np.square(actions).sum(0)
                        count += len(states)
                records.append(
                    {
                        "source": key,
                        "repo_id": entry["repo_id"],
                        "revision": entry["revision"],
                        "episode_index": eid,
                        "tasks": ep["tasks"],
                        "split": split,
                        "samples": n,
                        "path": relative,
                        "sha256": file_hash(path),
                    }
                )
                print(f"prepared {key} episode={eid} split={split} samples={n}", flush=True)
    if not count:
        raise ValueError("No training data selected.")
    stats = {}
    for key, sums, sums2 in [
        ("observation.state", sum_state, sum_state2),
        ("action", sum_action, sum_action2),
    ]:
        mean = sums / count
        std = np.sqrt(np.maximum(sums2 / count - mean**2, 0)).clip(1e-4)
        stats[key] = {"mean": mean.tolist(), "std": std.tolist()}
    write_json(out / "stats.json", stats)
    manifest = {
        "config": cfg,
        "signature": signature(reference),
        "episodes": records,
        "train_full_frames": count,
        "action_convention": cfg["action_convention"],
        "split_counts": {s: sum(r["split"] == s for r in records) for s in ["train", "val", "test"]},
        "normalization": "training episodes only, full frames, no repeated/padded chunk labels",
    }
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(out / "manifest.json", manifest)
    print(f"Prepared manifest: {out / 'manifest.json'}; splits={manifest['split_counts']}")
    return manifest
