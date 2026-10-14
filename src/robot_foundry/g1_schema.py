"""Explicit motion-only G1 VLA contract; separate from Panda and dexterous-hand G1 data."""

import numpy as np

from pathlib import Path

from .io import file_hash, fingerprint, read_json

ACTION_CONVENTION = "g1_sonic_v1_1_latent64"
ROBOT = "unitree_g1_sonic_motion29"
CAMERA = "observation.images.external"
STATE_DIM = 70
ACTION_DIM = 64
STATE_LAYOUT = {
    "joint_position_isaac": [0, 29],
    "joint_velocity_isaac": [29, 58],
    "base_angular_velocity": [58, 61],
    "projected_gravity": [61, 64],
    "base_linear_velocity_body": [64, 67],
    "pelvis_height": [67, 68],
    "heading_sin_cos": [68, 70],
}


def controller_identity(provenance):
    files = provenance["files"]
    hashes = {
        name: next(v["sha256"] for k, v in files.items() if k.endswith("/" + name))
        for name in ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]
    }
    return {"variant": "sonic_v1_1", **hashes, "fingerprint": fingerprint(hashes)}


def validate_g1_manifest(manifest, identity=None):
    if manifest.get("action_convention") != ACTION_CONVENTION:
        raise ValueError("Expected G1 SONIC latent64 data/checkpoint; Panda OSC actions are incompatible")
    signature = manifest["signature"]
    if signature["robot_type"] != ROBOT or signature["state"]["shape"] != [STATE_DIM]:
        raise ValueError("G1 VLA expects the explicit 70D proprioception layout")
    if signature["action"]["shape"] != [ACTION_DIM] or signature["cameras"] != [CAMERA]:
        raise ValueError("G1 motion VLA expects 64D SONIC tokens and one external camera")
    if signature["fps"] != 50 or manifest.get("state_layout") != STATE_LAYOUT:
        raise ValueError("G1 observation timing/layout differs from the deployed adapter")
    actual = manifest.get("controller_identity", {})
    if actual.get("variant") != "sonic_v1_1" or not actual.get("fingerprint"):
        raise ValueError("Missing exact SONIC controller identity")
    if identity is not None and actual != identity:
        raise ValueError("SONIC encoder/decoder/codebook differs from the training demonstrations")


def finite_token(token):
    value = np.asarray(token, np.float32)
    if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
        raise ValueError("Expected one finite 64D G1 SONIC token")
    return value


def load_g1_statistics(root, manifest):
    path = Path(root) / "stats.json"
    expected = manifest.get("statistics_sha256")
    if expected and file_hash(path) != expected:
        raise ValueError("G1 normalization statistics checksum differs from the training manifest")
    if not expected and manifest["config"].get("collection_version") == 1:
        raise ValueError("Versioned G1 collection requires a normalization statistics checksum")
    stats = read_json(path)
    for key, dimension in [("observation.state", STATE_DIM), ("action", ACTION_DIM)]:
        mean, std = (np.asarray(stats[key][field], np.float32) for field in ["mean", "std"])
        if mean.shape != (dimension,) or std.shape != (dimension,):
            raise ValueError(f"Invalid G1 normalization dimensions for {key}")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"Nonfinite/nonpositive G1 normalization statistics for {key}")
    return stats


def live_state(state):
    value = np.asarray(state, np.float32)
    if value.shape != (STATE_DIM,) or not np.isfinite(value).all():
        raise ValueError("Expected finite 70D current G1 state")
    if not np.isclose(np.linalg.norm(value[61:64]), 1, atol=0.02) or not np.isclose(
        np.linalg.norm(value[68:70]), 1, atol=0.02
    ):
        raise ValueError("G1 gravity direction and heading sin/cos must have unit norm")
    if value[67] <= 0:
        raise ValueError("G1 pelvis height must be positive")
    return value
