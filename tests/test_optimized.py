import os
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from robot_foundry.data import EpisodesDataset
from robot_foundry.distributed import StepBatchSampler
from robot_foundry.io import read_json, write_json
from robot_foundry.packed import build_packed_cache
from robot_foundry.train import train
from test_resume import make_data


def test_sampler_is_disjoint_and_resume_ignores_prefetch():
    options = dict(length=17, batch_size=4, accumulation=3, steps=5, seed=42, world_size=2)
    first = list(StepBatchSampler(**options, rank=0))
    second = list(StepBatchSampler(**options, rank=1))
    assert all(not set(a) & set(b) for a, b in zip(first, second))
    assert first[6:] == list(StepBatchSampler(**options, rank=0, start=2))
    assert second[6:] == list(StepBatchSampler(**options, rank=1, start=2))


def test_packed_cache_is_pixel_exact_and_normalization_unchanged(tmp_path):
    make_data(tmp_path)
    build_packed_cache(tmp_path)
    old = EpisodesDataset(tmp_path, "train")
    packed = EpisodesDataset(tmp_path, "train", packed=True)
    for i in range(len(old)):
        for key in ["images", "state", "action", "valid", "tokens"]:
            torch.testing.assert_close(old[i][key], packed[i][key], rtol=0, atol=0)
        assert old[i]["prompt"] == packed[i]["prompt"]


def config(root, output):
    return {
        "seed": 42,
        "backend": "tiny_bc",
        "prepared_dir": str(root),
        "device": "cpu",
        "training_engine": "ddp_amp_v2",
        "precision": "fp32",
        "steps": 4,
        "warmup_steps": 1,
        "batch_size": 1,
        "gradient_accumulation": 2,
        "lr": 3e-4,
        "eval_every": 2,
        "num_workers": 1,
        "output_dir": str(output),
        "verify_ddp_sync": True,
    }


def test_cuda_fp16_optimizer_and_scaler_checkpoint(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA AMP integration requires a real GPU")
    make_data(tmp_path)
    cfg = config(tmp_path, tmp_path / "cuda_run")
    cfg.update(device="cuda", precision="fp16", num_workers=0)
    path = tmp_path / "config.json"
    write_json(path, cfg)
    real_clip = torch.nn.utils.clip_grad_norm_
    calls = 0

    def inject_overflow(parameters, *args, **kwargs):
        nonlocal calls
        parameters = list(parameters)
        if calls == 0:
            parameters[0].grad.reshape(-1)[0] = float("inf")
        calls += 1
        return real_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", inject_overflow)
    report = train(path)
    state = torch.load(tmp_path / "cuda_run/training_state.pt", map_location="cpu", weights_only=False)
    assert report["precision"] == "fp16" and report["optimizer_updates"] == cfg["steps"] - 1
    assert report["amp_skipped_updates"] == 1
    assert state["scaler"]["scale"] == cfg.get("amp_init_scale", 1024) / 2
    assert state["scaler"] and np.isfinite(report["test"]["normalized_rmse"])


def run_ddp(path, log, *extra, check=True):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", USE_LIBUV="0", OMP_NUM_THREADS="2")
    if os.name == "nt":
        # Windows elastic c10d defaults to libuv, absent from this wheel. FileStore
        # still exercises real two-process Gloo/DDP; Linux CI uses actual torchrun below.
        store = log.parent / ("gloo_store_" + uuid.uuid4().hex)
        env["ROBOT_FOUNDRY_INIT_METHOD"] = store.as_uri()
        processes = []
        with log.open("w", encoding="utf-8") as stream:
            for rank in range(2):
                rank_env = dict(env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2")
                processes.append(
                    subprocess.Popen(
                        [sys.executable, str(Path(__file__).with_name("ddp_worker.py")), str(path), *extra],
                        env=rank_env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                )
            try:
                codes = [process.wait(timeout=120) for process in processes]
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=15)
        result = subprocess.CompletedProcess([], max(abs(c) for c in codes))
        if check and result.returncode:
            pytest.fail(log.read_text(encoding="utf-8")[-12000:])
        return result
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--rdzv-conf=use_libuv=0",
        "--nproc_per_node=2",
        str(Path(__file__).with_name("ddp_worker.py")),
        str(path),
        *extra,
    ]
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180, check=False
        )
    if check and result.returncode:
        pytest.fail(log.read_text(encoding="utf-8")[-12000:])
    return result


def test_two_process_ddp_sync_metrics_and_exact_resume(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch has no Gloo backend")
    root = tmp_path / "data"
    root.mkdir()
    make_data(root)
    # Uneven evaluation counts catch accidental duplicate padding and average-of-RMSE errors.
    with np.load(root / "val.npz") as arrays:
        values = {key: arrays[key][:5] for key in arrays.files}
    np.savez(root / "val.npz", **values)
    manifest = read_json(root / "manifest.json")
    manifest["episodes"][1]["samples"] = 5
    write_json(root / "manifest.json", manifest)
    full = tmp_path / "full.json"
    resumed = tmp_path / "resumed.json"
    write_json(full, config(root, tmp_path / "full"))
    write_json(resumed, config(root, tmp_path / "resumed"))
    run_ddp(full, tmp_path / "full.log")
    interrupted = run_ddp(resumed, tmp_path / "interrupted.log", "--interrupt-at", "2", check=False)
    assert interrupted.returncode != 0
    run_ddp(resumed, tmp_path / "resumed.log", "--resume")
    a = torch.load(tmp_path / "full/training_state.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path / "resumed/training_state.pt", map_location="cpu", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    assert len(b["rank_states"]) == b["world_size"] == 2
    report = read_json(tmp_path / "resumed/report.json")
    assert report["effective_batch"] == 4
    assert len(set(report["parameter_digests_by_rank"])) == 1
    assert len(set(report["training_parameter_digests_by_rank"])) == 1
    history = [
        __import__("json").loads(line)
        for line in (tmp_path / "resumed/metrics.jsonl").read_text().splitlines()
    ]
    assert all(row["val"]["evaluated_samples"] == 5 for row in history)
    assert report["test"]["evaluated_samples"] == 6
    # Compare sufficient-statistic reduction against a single-process evaluation.
    from robot_foundry.backends import make_backend
    from robot_foundry.distributed import Runtime
    from robot_foundry.train_optimized import evaluate

    single_cfg = config(root, tmp_path / "unused")
    dataset = EpisodesDataset(root, "test")
    backend = make_backend(single_cfg, dataset, torch.device("cpu"))
    selected = torch.load(tmp_path / "resumed/best/model.pt", weights_only=True)
    backend.model.load_state_dict(selected["model"])
    single = evaluate(backend, dataset, single_cfg, Runtime(single_cfg), split="test")
    assert report["test"]["normalized_rmse"] == pytest.approx(single["normalized_rmse"], abs=1e-8)


def test_parameter_digest_handles_bfloat16_and_scalar_buffers():
    from robot_foundry.train_optimized import parameter_digest

    model = torch.nn.Linear(2, 2).bfloat16()
    model.register_buffer("scalar", torch.tensor(3.0, dtype=torch.bfloat16))
    before = parameter_digest(model)
    model.scalar.add_(1)
    assert len(before) == 64 and parameter_digest(model) != before
