import av
import numpy as np
import pytest
import torch

from robot_foundry.data import masked_loss, tokenize
from robot_foundry.datasets import split_episodes
from robot_foundry.model import TinyChunkPolicy
from robot_foundry.prepare import action_chunk
from robot_foundry.simulation import decode_libero_action, quat_xyzw_to_axis_angle, wilson
from robot_foundry.video import decode_frames


def test_split_never_leaks_episode_and_is_stratified():
    eps = [{"episode_index": i, "tasks": [f"task_{i // 10}"]} for i in range(40)]
    a = split_episodes(eps, 42, per_task=7)
    b = split_episodes(eps, 42, per_task=7)
    assert a == b
    sets = [{e["episode_index"] for e in a[k]} for k in ["train", "val", "test"]]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert all({e["tasks"][0] for e in rows} == {f"task_{i}" for i in range(4)} for rows in a.values())


def test_tail_padding_has_zero_loss_gradient_and_cannot_cross_episode():
    actions = np.asarray([[1, 2], [3, 4]], np.float32)
    chunk, valid = action_chunk(actions, 1, 4)
    assert chunk.tolist() == [[3, 4]] * 4
    assert valid.tolist() == [True, False, False, False]
    pred = torch.zeros(1, 4, 2, requires_grad=True)
    loss = masked_loss(pred, torch.tensor(chunk).unsqueeze(0), torch.tensor(valid).unsqueeze(0))
    loss.backward()
    assert torch.count_nonzero(pred.grad[:, 1:]) == 0


def test_video_timestamp_offsets_and_truncation(tmp_path):
    path = tmp_path / "test.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 15, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    arr = decode_frames(path, [0.3, 0.45, 0.55], 32, 20)
    assert arr.shape == (3, 3, 32, 32)
    assert np.allclose(arr.mean((1, 2, 3)), [90, 135, 165], atol=4)
    with pytest.raises(ValueError, match="truncated"):
        decode_frames(path, [1.0], 32, 20)


def test_real_backprop_is_finite_and_output_dimensions_match():
    torch.set_num_threads(2)
    policy = TinyChunkPolicy(8, 7, 2, 8)
    batch = {
        "images": torch.randint(255, (2, 2, 3, 32, 32), dtype=torch.uint8),
        "state": torch.randn(2, 8),
        "tokens": torch.tensor(np.stack([tokenize("pick a can")] * 2)),
    }
    out = policy(batch)
    assert out.shape == (2, 8, 7)
    out.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())


def test_libero_gripper_convention_does_not_mutate_input():
    a = np.zeros(7)
    a[-1] = 1
    assert decode_libero_action(a)[-1] == -1
    assert a[-1] == 1
    a[-1] = 0
    assert decode_libero_action(a)[-1] == 1
    with pytest.raises(ValueError):
        decode_libero_action(np.full(7, np.nan))


def test_rotation_identity_and_pi():
    np.testing.assert_allclose(quat_xyzw_to_axis_angle([0, 0, 0, 1]), [0, 0, 0])
    np.testing.assert_allclose(quat_xyzw_to_axis_angle([0, 0, 1, 0]), [0, 0, np.pi], atol=1e-6)


def test_confidence_interval_including_boundary_cases():
    low, high = wilson(0, 50)
    assert 0 <= low < high < 0.1
    low, high = wilson(50, 50)
    assert 0.9 < low < high <= 1
