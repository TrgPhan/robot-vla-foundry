import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_foundry.g1_schema import (
    ACTION_CONVENTION,
    CAMERA,
    ROBOT,
    STATE_LAYOUT,
    controller_identity,
    live_state,
    load_g1_statistics,
    validate_g1_manifest,
)
from robot_foundry.g1_transport import decode_request, encode_request
from robot_foundry.io import file_hash, write_json
from robot_foundry.smol_adaptation import adapt_g1_boundaries


def physical_state():
    state = np.zeros(70, np.float32)
    state[63], state[67], state[69] = -1, 0.79, 1
    return state


def test_g1_projection_expansion_keeps_pretrained_expert_and_trains_new_heads():
    core = torch.nn.Module()
    core.state_proj = torch.nn.Linear(32, 6)
    core.action_in_proj = torch.nn.Linear(32, 4)
    core.action_out_proj = torch.nn.Linear(4, 32)
    core.expert = torch.nn.Linear(4, 4)
    expert = core.expert.weight
    policy = SimpleNamespace(model=core, config=SimpleNamespace(max_state_dim=32, max_action_dim=32))
    meta = adapt_g1_boundaries(policy)
    assert meta["max_state_dim"] == 96 and meta["max_action_dim"] == 64
    assert core.expert.weight is expert
    assert core.state_proj.in_features == 96 and core.action_out_proj.out_features == 64
    before = core.action_out_proj.weight.detach().clone()
    optimizer = torch.optim.SGD(core.parameters(), lr=0.01)
    loss = core.state_proj(torch.ones(1, 96)).square().mean()
    loss += core.action_out_proj(core.action_in_proj(torch.ones(1, 64))).square().mean()
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, core.action_out_proj.weight)


def test_g1_rpc_roundtrip_and_physical_state_validation():
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    state, decoded, prompt = decode_request(encode_request(physical_state(), image, "Walk."), 32)
    np.testing.assert_array_equal(decoded, image)
    np.testing.assert_array_equal(state, physical_state())
    assert prompt == "Walk."
    with pytest.raises(ValueError, match="unit norm"):
        live_state(np.zeros(70))


def test_g1_normalization_rejects_changed_or_nonphysical_statistics(tmp_path):
    from robot_foundry.backends import TinyBackend

    stats = {k: {"mean": [0.0] * d, "std": [1.0] * d} for k, d in [("observation.state", 70), ("action", 64)]}
    path = tmp_path / "stats.json"
    path.write_bytes(json.dumps(stats, separators=(",", ":")).encode() + b"\r\n")
    manifest = {
        "config": {"collection_version": 1, "horizon": 2},
        "signature": {"state": {"shape": [70]}, "action": {"shape": [64]}, "cameras": [CAMERA]},
        "statistics_sha256": file_hash(path),
    }
    assert load_g1_statistics(tmp_path, manifest) == stats
    dataset = SimpleNamespace(root=tmp_path, manifest=manifest)
    TinyBackend({}, dataset, torch.device("cpu")).export(tmp_path / "best", dataset)
    assert file_hash(tmp_path / "best/stats.json") == manifest["statistics_sha256"]
    stats["action"]["mean"][0] = 42.0
    write_json(path, stats)
    with pytest.raises(ValueError, match="checksum"):
        load_g1_statistics(tmp_path, manifest)
    stats["action"]["std"][0] = 0.0
    write_json(path, stats)
    manifest["statistics_sha256"] = file_hash(path)
    with pytest.raises(ValueError, match="nonpositive"):
        load_g1_statistics(tmp_path, manifest)


def test_g1_collection_pairs_pre_action_observations_with_their_latent_targets(tmp_path, monkeypatch):
    import robot_foundry.g1_collect as collection

    class Env:
        def __init__(self, *args):
            self.value = 0

        def state(self):
            result = physical_state()
            result[0] = self.value
            return result

        def image(self):
            return np.full((32, 32, 3), self.value, np.uint8)

        def proprio(self):
            return (self.value,)

        def step(self, action):
            self.value += 1
            return {"fall": False, "height_m": 0.79, "tilt_deg": 0}

        def close(self):
            pass

    class Teacher:
        def __init__(self, models):
            pass

        def predict(self, value, motion, frame, playing=True):
            return np.zeros(29), np.full(64, 10 + value, np.float32)

    class Video:
        def __init__(self, *args):
            pass

        def write(self, *args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(collection, "G1Environment", Env)
    monkeypatch.setattr(collection, "SonicController", Teacher)
    monkeypatch.setattr(collection, "CameraVideo", Video)
    monkeypatch.setattr(collection, "scenario_motion", lambda x, seed: x)
    cfg = {"image_size": 32, "max_frames": 3, "warmup_frames": 0, "stride": 1, "horizon": 2}
    arrays, _ = collection.collect_episode(
        None, {"joint_pos": np.zeros((3, 29))}, None, 42, cfg, tmp_path / "test.mp4"
    )
    np.testing.assert_array_equal(arrays["states"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(arrays["images"][:, 0, 0, 0, 0], [0, 1, 2])
    np.testing.assert_array_equal(arrays["actions"][:, 0, 0], [10, 11, 12])
    np.testing.assert_array_equal(arrays["valid"][-1], [True, False])


def test_learned_rollout_uses_only_policy_tokens_after_initialization(tmp_path, monkeypatch):
    import robot_foundry.g1_rollout as rollout

    provenance = {
        "files": {
            f"policy/{n}": {"sha256": "a" * 64}
            for n in ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]
        }
    }
    identity = controller_identity(provenance)
    cfg = {
        "horizon": 4,
        "image_size": 32,
        "warmup_frames": 2,
        "repo": "repo",
        "assets": "assets",
        "tasks": [{"motion": "fixture", "prompt": "Walk."}],
    }
    manifest = {
        "config": cfg,
        "signature": {
            "robot_type": ROBOT,
            "fps": 50,
            "cameras": [CAMERA],
            "state": {"shape": [70]},
            "action": {"shape": [64]},
        },
        "state_layout": STATE_LAYOUT,
        "action_convention": ACTION_CONVENTION,
        "controller_identity": identity,
        "episodes": [{"task_id": 0, "split": "test", "seed": 73, "length": 3}],
        "scope": "test fixture",
        "benchmark_scope": "unit test",
        "fingerprint": "fixture",
    }
    checkpoint = tmp_path / "best"
    write_json(checkpoint / "data_manifest.json", manifest)
    write_json(
        checkpoint / "stats.json",
        {k: {"mean": [0.0] * d, "std": [1.0] * d} for k, d in [("observation.state", 70), ("action", 64)]},
    )
    (checkpoint / "model.safetensors").write_bytes(b"fixture-only")
    counters = {"initialization": 0, "learned": 0}

    class Policy:
        horizon = 4

        def __init__(self, *args):
            self.health = {
                "backend": "smolvla",
                "optimizer_updates": 1,
                "weights_sha256": file_hash(checkpoint / "model.safetensors"),
            }

        def predict(self, *args):
            return np.ones((4, 64), np.float32)

    class Controller:
        reference_encoder_calls = 0

        def __init__(self, models):
            pass

        def predict(self, *args, **kwargs):
            assert counters["initialization"] < 2, "Forbidden teacher fallback during learned control"
            counters["initialization"] += 1
            self.reference_encoder_calls += 1
            return np.zeros(29), np.zeros(64)

        def decode_token(self, *args):
            np.testing.assert_array_equal(args[-1], np.ones(64))
            counters["learned"] += 1
            return np.zeros(29)

    class Env:
        def __init__(self, *args, **kwargs):
            qpos = np.zeros(36)
            qpos[2] = 0.79
            qpos[3] = 1
            self.data = SimpleNamespace(qpos=qpos)

        def proprio(self):
            return ()

        def state(self):
            return physical_state()

        def image(self):
            return np.zeros((32, 32, 3), np.uint8)

        def render(self):
            return np.zeros((512, 512, 3), np.uint8)

        def step(self, action):
            return {"fall": False, "height_m": 0.79, "tilt_deg": 0}

        def close(self):
            pass

    class Video:
        def __init__(self, *args):
            pass

        def write(self, *args):
            pass

        def close(self):
            pass

    motion = {"joint_pos": np.zeros((3, 29)), "body_quat": np.tile([1, 0, 0, 0], (3, 1, 1))}
    monkeypatch.setattr(rollout, "RemoteG1Policy", Policy)
    monkeypatch.setattr(rollout, "SonicController", Controller)
    monkeypatch.setattr(rollout, "G1Environment", Env)
    monkeypatch.setattr(rollout, "CameraVideo", Video)
    monkeypatch.setattr(rollout, "load_motion", lambda p: motion)
    monkeypatch.setattr(rollout, "scenario_motion", lambda m, s: m)
    monkeypatch.setattr(rollout, "prepare_sonic_assets", lambda *a: (None, None, None, provenance))
    result = rollout.evaluate_g1(checkpoint, "http://localhost", tmp_path / "evaluation")
    assert counters == {"initialization": 2, "learned": 3}
    assert result["teacher_calls_after_initialization"] == 0
    assert result["vla_used"] and result["per_trial"][0]["learned_tokens_executed"] == 3
    incompatible = dict(manifest, controller_identity={"variant": "other"})
    with pytest.raises(ValueError, match="identity"):
        validate_g1_manifest(incompatible)


def test_primary_notebook_contains_training_and_trained_g1_inference():
    notebook = json.loads(Path("notebooks/kaggle_train.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "scripts/run_g1_train.py" in code and "scripts/run_g1_vla_sim.py" in code
    assert "scripts/run_fullbody_sim.py" not in code
    for i, c in enumerate(notebook["cells"]):
        if c["cell_type"] == "code":
            compile("".join(c["source"]), f"cell_{i}", "exec")
