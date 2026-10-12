"""Lossless random-access PNG episode caches to avoid decompressing a whole NPZ per sample."""

import io
import os
from pathlib import Path

import numpy as np
from filelock import FileLock
from PIL import Image

from .io import file_hash, read_json, write_json

CACHE_VERSION = 1


def cache_path(root, record):
    digest = record.get("sha256") or file_hash(Path(root) / record["path"])
    return Path(root) / "packed" / f"{Path(record['path']).stem}_{digest[:16]}_v{CACHE_VERSION}"


def build_episode(root, record):
    target = cache_path(root, record)
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock"):
        if (target / "ready.json").exists():
            return target
        target.mkdir(exist_ok=True)
        with np.load(Path(root) / record["path"], allow_pickle=False) as arrays:
            images = arrays["images"]
            offsets = [0]
            temp_blob = target / "images.tmp"
            with temp_blob.open("wb") as blob:
                for image in images.reshape(-1, *images.shape[2:]):
                    buffer = io.BytesIO()
                    Image.fromarray(image.transpose(1, 2, 0)).save(buffer, format="PNG", compress_level=1)
                    pixels = buffer.getvalue()
                    blob.write(pixels)
                    offsets.append(offsets[-1] + len(pixels))
            os.replace(temp_blob, target / "images.bin")
            for key in ["states", "actions", "valid", "prompts"]:
                temp = target / f"{key}.tmp.npy"
                np.save(temp, arrays[key], allow_pickle=False)
                os.replace(temp, target / f"{key}.npy")
            np.save(target / "offsets.npy", np.asarray(offsets, dtype=np.int64), allow_pickle=False)
            write_json(target / "ready.json", {"version": CACHE_VERSION, "image_shape": list(images.shape)})
    return target


def build_packed_cache(root):
    manifest = read_json(Path(root) / "manifest.json")
    for i, record in enumerate(manifest["episodes"]):
        build_episode(root, record)
        if (i + 1) % 20 == 0:
            print(f"Packed lossless image cache: {i + 1}/{len(manifest['episodes'])} episodes", flush=True)


def open_episode(root, record):
    target = cache_path(root, record)
    if not (target / "ready.json").exists():
        raise FileNotFoundError("Packed data cache missing. Run the Kaggle launcher preparation first.")
    result = {
        key: np.load(target / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in ["states", "actions", "valid", "prompts"]
    }
    result["image_shape"] = read_json(target / "ready.json")["image_shape"]
    result["image_blob"] = np.memmap(target / "images.bin", dtype=np.uint8, mode="r")
    result["offsets"] = np.load(target / "offsets.npy", mmap_mode="r", allow_pickle=False)
    return result


def image_row(episode, row):
    views = episode["image_shape"][1]
    images = []
    for view in range(views):
        index = row * views + view
        low, high = episode["offsets"][index : index + 2]
        with Image.open(io.BytesIO(episode["image_blob"][low:high].tobytes())) as image:
            images.append(np.asarray(image.convert("RGB")).transpose(2, 0, 1).copy())
    return np.stack(images)
