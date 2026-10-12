import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_foundry.data import EpisodesDataset
from robot_foundry.io import file_hash, fingerprint, read_json, write_json
from robot_foundry.packed import build_packed_cache, cache_path, compact_episode
from robot_foundry.storage import (
    archive_directory,
    atomic_export,
    recover_export,
    recover_temporary_checkpoint,
)
from robot_foundry.train import atomic_checkpoint, train
from test_optimized import config
from test_resume import make_data


def verified_data(root):
    make_data(root)
    manifest = read_json(root / "manifest.json")
    for record in manifest["episodes"]:
        record["sha256"] = file_hash(root / record["path"])
    manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
    write_json(root / "manifest.json", manifest)
    return manifest


def test_compaction_preserves_samples_and_manifest_without_npz(tmp_path):
    manifest = verified_data(tmp_path)
    before = [EpisodesDataset(tmp_path, "train")[i] for i in range(6)]
    freed = build_packed_cache(tmp_path, compact=True)
    assert freed > 0
    assert all(not (tmp_path / r["path"]).exists() for r in manifest["episodes"])
    assert read_json(tmp_path / "manifest.json") == manifest
    for packed in [False, True]:
        after = EpisodesDataset(tmp_path, "train", packed=packed)
        for i, old in enumerate(before):
            for key in ["images", "state", "action", "valid", "tokens"]:
                torch.testing.assert_close(after[i][key], old[key], rtol=0, atol=0)
    assert build_packed_cache(tmp_path, compact=True) == 0


def test_corrupt_cache_never_deletes_original(tmp_path):
    manifest = verified_data(tmp_path)
    build_packed_cache(tmp_path)
    record = manifest["episodes"][0]
    blob = cache_path(tmp_path, record) / "images.bin"
    with blob.open("r+b") as stream:
        stream.write(b"invalid!")
    with pytest.raises((ValueError, OSError)):
        compact_episode(tmp_path, record)
    assert file_hash(tmp_path / record["path"]) == record["sha256"]


def test_failed_model_export_keeps_previous_best_and_frees_partial(tmp_path):
    best = tmp_path / "best"
    best.mkdir()
    (best / "model").write_bytes(b"previous-good-model")

    class Broken:
        def export(self, stage, dataset, asset_destination=None):
            (stage / "model").write_bytes(b"partial-new-model")
            raise OSError(28, "No space left on device")

    with pytest.raises(OSError, match="No space"):
        atomic_export(Broken(), best, None, {"step": 2})
    assert (best / "model").read_bytes() == b"previous-good-model"
    assert not (tmp_path / ".best.pending").exists()

    class Good:
        def export(self, stage, dataset, asset_destination=None):
            (stage / "model").write_bytes(b"new-good-model")
            write_json(stage / "asset_path.json", str(asset_destination))

    atomic_export(Good(), best, None, {"step": 2})
    assert (best / "model").read_bytes() == b"new-good-model"
    assert read_json(best / "asset_path.json") == str(best / "vlm_assets")
    assert not (tmp_path / ".best.previous").exists()
    best.rename(tmp_path / ".best.previous")
    recover_export(best)
    assert (best / "model").read_bytes() == b"new-good-model"


def test_failed_optimizer_save_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "training_state.pt"
    atomic_checkpoint(path, {"step": 1})

    def failed_save(value, target):
        target.write_bytes(b"partial")
        raise RuntimeError("disk write failed")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", failed_save)
        with pytest.raises(RuntimeError, match="disk write"):
            atomic_checkpoint(path, {"step": 2})
    assert torch.load(path, weights_only=True)["step"] == 1
    assert not path.with_suffix(".tmp").exists()


def test_complete_temporary_checkpoint_is_salvaged(tmp_path):
    path = tmp_path / "training_state.pt"
    value = {
        "engine": "ddp_amp_v2",
        "config_hash": "recipe",
        "data_fingerprint": "data",
        "world_size": 2,
        "model": {},
        "optimizer": {},
        "scheduler": {},
        "scaler": {},
        "rank_states": [],
        "step": 1,
    }
    torch.save(value, path)
    torch.save(dict(value, step=2), path.with_suffix(".tmp"))
    assert recover_temporary_checkpoint(path, "recipe", "data") == "promoted-complete"
    assert torch.load(path, weights_only=True)["step"] == 2
    torch.save(dict(value, config_hash="different"), path.with_suffix(".tmp"))
    with pytest.raises(ValueError, match="different recipe"):
        recover_temporary_checkpoint(path, "recipe", "data")
    assert path.with_suffix(".tmp").exists()


def test_disk_guard_aborts_before_training(tmp_path, monkeypatch):
    make_data(tmp_path)
    cfg = config(tmp_path, tmp_path / "run")
    cfg.update(num_workers=0)
    path = tmp_path / "config.json"
    write_json(path, cfg)
    monkeypatch.setattr("robot_foundry.storage.shutil.disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="Insufficient disk"):
        train(path)
    assert not (tmp_path / "run/training_state.pt").exists()


def test_archive_failure_preserves_previous_zip_and_removes_partial(tmp_path, monkeypatch):
    import zipfile

    source = tmp_path / "outputs/run"
    source.mkdir(parents=True)
    (source / "model").write_bytes(b"model")
    destination = tmp_path / "results.zip"
    archive_directory(source, destination, tmp_path)
    previous = destination.read_bytes()

    def fail(*args, **kwargs):
        raise OSError(28, "No space")

    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile, "write", fail)
        with pytest.raises(OSError):
            archive_directory(source, destination, tmp_path)
    assert destination.read_bytes() == previous
    assert not destination.with_suffix(".zip.tmp").exists()


def test_resume_with_incomplete_best_rescores_durable_checkpoint(tmp_path, monkeypatch):
    make_data(tmp_path)
    cfg = config(tmp_path, tmp_path / "run")
    cfg.update(num_workers=0)
    path = tmp_path / "config.json"
    write_json(path, cfg)
    module = importlib.import_module("robot_foundry.train_optimized")
    original = module.atomic_checkpoint

    def stop(target, value):
        original(target, value)
        raise InterruptedError("stopped after durable checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_checkpoint", stop)
        with pytest.raises(InterruptedError):
            train(path)
    (tmp_path / "run/best/model.pt").write_bytes(b"truncated old export")
    report = train(path, resume=True)
    assert report["optimizer_updates"] == cfg["steps"]
    assert report["best_reconstructed_from_resume"]
    assert np.isfinite(report["test"]["normalized_rmse"])


def test_compact_preparation_can_reenter_without_decode_or_split_change(tmp_path, monkeypatch):
    module = importlib.import_module("robot_foundry.prepare")
    raw = tmp_path / "raw/source"
    raw.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "fps": 20,
        "chunks_size": 1000,
        "data_path": "episode_{episode_index}.parquet",
        "video_path": "episode_{episode_index}.mp4",
        "features": {
            "observation.state": {"shape": [8], "dtype": "float32"},
            "action": {"shape": [7], "dtype": "float32"},
            "camera": {"dtype": "video"},
        },
    }
    episodes = [{"episode_index": i, "tasks": ["task"], "length": 6} for i in range(3)]
    for ep in episodes:
        for ext in ["parquet", "mp4"]:
            (raw / f"episode_{ep['episode_index']}.{ext}").write_bytes(b"test-source")
    registry = tmp_path / "registry.json"
    write_json(
        registry, {"source": {"repo_id": "test/source", "revision": "test", "action_convention": "test"}}
    )
    cfg = {
        "registry": str(registry),
        "datasets": ["source"],
        "raw_dir": str(tmp_path / "raw"),
        "prepared_dir": str(tmp_path / "prepared"),
        "seed": 42,
        "horizon": 4,
        "stride": 2,
        "image_size": 32,
        "action_convention": "test",
        "packed_cache": True,
    }
    path = tmp_path / "config.json"
    write_json(path, cfg)
    monkeypatch.setattr(module, "metadata", lambda *a: (info, episodes, {0: "task"}))

    def load(entry, root, info, ep, cameras):
        return (
            np.full((6, 8), ep["episode_index"], np.float32),
            np.ones((6, 7), np.float32),
            np.arange(6) / 20,
            np.zeros(6, np.int64),
            {"camera": (f"episode_{ep['episode_index']}.mp4", 0)},
        )

    monkeypatch.setattr(module, "load_episode", load)
    monkeypatch.setattr(
        module, "decode_frames", lambda path, times, *a: np.zeros((len(times), 3, 32, 32), np.uint8)
    )
    first = module.prepare(path, compact=True)
    stats = read_json(tmp_path / "prepared/stats.json")
    monkeypatch.setattr(module, "load_episode", lambda *a: pytest.fail("Unexpected raw redownload"))
    monkeypatch.setattr(module, "decode_frames", lambda *a: pytest.fail("Unexpected video redecode"))
    second = module.prepare(path, compact=True)
    assert first == second and read_json(tmp_path / "prepared/stats.json") == stats
