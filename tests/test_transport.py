import numpy as np
import pytest

from robot_foundry.transport import RemotePredictor, decode_observation, encode_observation


def test_exact_rgb_transport_and_state_validation():
    rng = np.random.default_rng(42)
    obs = {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.asarray([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.zeros(2),
        "agentview_image": rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
    }
    payload = encode_observation(obs, "pick the milk")
    decoded, prompt = decode_observation(payload)
    assert prompt == "pick the milk"
    for key, value in obs.items():
        np.testing.assert_array_equal(decoded[key], value)
    payload["state"]["robot0_eef_quat"] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="quaternion"):
        decode_observation(payload)


def test_transport_is_local_only():
    with pytest.raises(ValueError, match="loopback"):
        RemotePredictor("https://example.com", {"config": {"horizon": 16}})
