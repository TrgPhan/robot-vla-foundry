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


def validate_cache(root, record):
    """Check an existing packed cache before freeing its original NPZ."""
    target = cache_path(root, record)
    ready = read_json(target / "ready.json")
    episode = open_episode(root, record)
    shape = ready["image_shape"]
    if shape[0] != record["samples"]:
        raise ValueError(f"Packed sample count changed: {target}")
    offsets = episode["offsets"]
    if (
        len(offsets) != shape[0] * shape[1] + 1
        or offsets[0] != 0
        or offsets[-1] != len(episode["image_blob"])
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError(f"Incomplete packed image blob: {target}")
    for name, digest in ready.get("file_hashes", {}).items():
        if Path(name).name != name or file_hash(target / name) != digest:
            raise ValueError(f"Packed cache checksum mismatch: {target / name}")
    return target


def compact_episode(root, record):
    """Delete only a byte-verified NPZ after proving its packed pixels/labels are equivalent."""
    source = Path(root) / record["path"]
    root = Path(root).resolve()
    if not source.resolve().is_relative_to(root) or source.is_symlink():
        raise ValueError("Prepared NPZ must be a regular file inside its declared root.")
    if not source.exists():
        validate_cache(root, record)
        return 0
    if not record.get("sha256") or file_hash(source) != record["sha256"]:
        raise ValueError(f"Refusing to compact an unverified NPZ: {source}")
    target = validate_cache(root, record)
    episode = open_episode(root, record)
    with np.load(source, allow_pickle=False) as original:
        for key in ["states", "actions", "valid", "prompts"]:
            if not np.array_equal(original[key], episode[key]):
                raise ValueError(f"Packed {key} differ: {source}")
        images = original["images"]
        if tuple(images.shape) != tuple(episode["image_shape"]):
            raise ValueError(f"Packed image shape differs: {source}")
        for row in range(len(images)):
            if not np.array_equal(images[row], image_row(episode, row)):
                raise ValueError(f"Packed RGB differs at sample {row}: {source}")
        # Retain full-frame statistics inputs for interrupted preparation/reuse.
        for key in ["full_states", "full_actions"]:
            if key in original:
                temp = target / f"{key}.tmp.npy"
                np.save(temp, original[key], allow_pickle=False)
                os.replace(temp, target / f"{key}.npy")
    files = ["images.bin", "offsets.npy", "states.npy", "actions.npy", "valid.npy", "prompts.npy"]
    files += [k + ".npy" for k in ["full_states", "full_actions"] if (target / (k + ".npy")).exists()]
    ready = read_json(target / "ready.json")
    ready.update(
        source_sha256=record["sha256"], file_hashes={name: file_hash(target / name) for name in files}
    )
    write_json(target / "ready.json", ready)
    # Persist the lookup before unlink: a killed prepare can resume without redownloading.
    meta = source.with_suffix(".json")
    provenance = read_json(meta) if meta.exists() else {}
    provenance["packed_record"] = record
    write_json(meta, provenance)
    size = source.stat().st_size
    source.unlink()
    return size


def build_packed_cache(root, compact=False):
    manifest = read_json(Path(root) / "manifest.json")
    freed = 0
    for i, record in enumerate(manifest["episodes"]):
        build_episode(root, record)
        if compact:
            freed += compact_episode(root, record)
        if (i + 1) % 20 == 0:
            print(f"Packed lossless image cache: {i + 1}/{len(manifest['episodes'])} episodes", flush=True)
    if compact:
        print(f"Verified compact data: freed {freed / 2**30:.2f} GiB of duplicate NPZ files", flush=True)
    return freed


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
