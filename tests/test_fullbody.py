from collections import deque

import numpy as np
import pytest

from robot_foundry.fullbody import (
    DEFAULT_MJ,
    ISAAC_FOR_MJ,
    MJ_FOR_ISAAC,
    SonicController,
    heading_rotation,
    motor_parameters,
    require_g1_vla_checkpoint,
    rotation_wxyz,
)
from robot_foundry.io import write_json


def test_panda_checkpoint_cannot_be_used_as_whole_body_vla(tmp_path):
    write_json(
        tmp_path / "data_manifest.json",
        {
            "action_convention": "libero_osc_delta_xyz_axis_angle_gripper_open01",
            "signature": {"action": {"shape": [7]}},
        },
    )
    with pytest.raises(ValueError, match="Panda 7D OSC"):
        require_g1_vla_checkpoint(tmp_path)


def test_heading_removes_yaw_but_keeps_physical_tilt():
    # WXYZ: ninety-degree yaw followed by thirty-degree roll.
    yaw = np.array([2**-0.5, 0, 0, 2**-0.5])
    matrix = rotation_wxyz(yaw)
    np.testing.assert_allclose(matrix @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(heading_rotation(yaw).T @ matrix, np.eye(3), atol=1e-12)
    roll = rotation_wxyz([np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0])
    np.testing.assert_allclose(roll.T @ [0, 0, -1], [0, -0.5, -(3**0.5) / 2], atol=1e-12)
    with pytest.raises(ValueError, match="nonzero"):
        rotation_wxyz([0, 0, 0, 0])


def test_policy_to_motor_contract_roundtrip_and_gains():
    np.testing.assert_array_equal(MJ_FOR_ISAAC[ISAAC_FOR_MJ], np.arange(29))
    # An action for Isaac left knee affects hardware left knee, not a hip or right leg.
    action = np.zeros(29)
    action[9] = 1
    kp, kd, scale = motor_parameters()
    delta = action[ISAAC_FOR_MJ] * scale
    assert np.flatnonzero(delta).tolist() == [3]
    assert kp[3] == pytest.approx(0.025101925 * (20 * np.pi) ** 2)
    assert scale[3] * kp[3] == pytest.approx(0.25 * 139)
    assert kp[4] == pytest.approx(3 * 0.003609725 * (20 * np.pi) ** 2)
    assert kd[4] == pytest.approx(12 * 0.003609725 * 20 * np.pi)


def test_encoder_future_window_and_decoder_chronological_history():
    # Capture ONNX boundary tensors; known reference contract from NVIDIA C++ gatherers.
    controller = object.__new__(SonicController)
    from robot_foundry.fullbody import DECODER_DIMS, ENCODER_DIMS

    controller.encoder_names, controller.decoder_names = list(ENCODER_DIMS), list(DECODER_DIMS)
    controller.history = deque(maxlen=10)
    for i in range(9):
        controller.history.append(
            {
                "ang": np.ones(3) * i,
                "q": np.zeros(29),
                "dq": np.zeros(29),
                "action": np.zeros(29),
                "gravity": np.array([0, 0, -1]),
            }
        )
    controller.last_action = np.arange(29)
    controller.encoder, controller.decoder = "encoder", "decoder"
    captured = {}

    def infer(session, obs):
        captured[session] = obs
        return np.zeros(64 if session == "encoder" else 29)

    controller._infer = infer
    motion = {
        "joint_pos": np.repeat(np.arange(12)[:, None], 29, axis=1),
        "joint_vel": np.ones((12, 29)),
        "body_quat": np.tile([1, 0, 0, 0], (12, 1, 1)),
    }
    controller.predict(DEFAULT_MJ.copy(), np.zeros(29), [1, 0, 0, 0], np.ones(3) * 9, motion, 2)
    assert captured["encoder"].shape == (1751,)
    assert captured["decoder"].shape == (994,)
    np.testing.assert_array_equal(captured["encoder"][:4], [0, 0, 0, 0])  # scalar mode zero
    np.testing.assert_array_equal(
        captured["encoder"][4:294].reshape(10, 29)[:, 0], [2, 7, 11, 11, 11, 11, 11, 11, 11, 11]
    )
    np.testing.assert_array_equal(
        captured["encoder"][584:644].reshape(10, 6), np.tile([1, 0, 0, 1, 0, 0], (10, 1))
    )
    np.testing.assert_array_equal(captured["decoder"][64:94].reshape(10, 3)[:, 0], np.arange(10))
    np.testing.assert_array_equal(captured["decoder"][94:384], np.zeros(290))
