import ast
import os
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from robot_foundry.io import read_json, write_json
from robot_foundry.native import groot_command, validate_g1_dataset


def test_groot_flags_match_pinned_reference_dataclass():
    reference = Path("vendor/groot/gr00t/configs/finetune_config.py")
    if not reference.exists():
        pytest.skip("Optional pinned GR00T checkout is not bootstrapped")
    tree = ast.parse(reference.read_text(encoding="utf-8"))
    fields = {
        node.target.id.replace("_", "-")
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    cfg = read_json("configs/groot_libero.json")
    cmd = groot_command(cfg, "vendor/groot", ["first", "second"], "model")
    flags = {item[2:] for item in cmd if item.startswith("--")}
    assert flags <= fields
    assert cmd[cmd.index("--dataset-path") + 1] == os.pathsep.join(
        str(Path(p).resolve()) for p in ["first", "second"]
    )
    assert cfg["global_batch_size"] * cfg["gradient_accumulation_steps"] == 32


def test_no_fake_sonic_tokens_for_arm_dataset(tmp_path):
    write_json(tmp_path / "meta/info.json", {"codebase_version": "v2.1", "fps": 50})
    write_json(tmp_path / "meta/modality.json", {"action": {"motion_token": {"start": 0, "end": 7}}})
    with pytest.raises(ValueError, match="64"):
        validate_g1_dataset(tmp_path)


@pytest.fixture
def g1_schema_fixture(tmp_path):
    # Synthetic schema fixture only; not demonstrations or an evaluated controller.
    features = {
        "action": {"shape": [78]},
        "observation.state": {"shape": [43]},
        "observation.projected_gravity": {"shape": [3]},
    }
    action = {
        "motion_token": {"start": 0, "end": 64},
        "left_hand_joints": {"start": 64, "end": 71},
        "right_hand_joints": {"start": 71, "end": 78},
    }
    state = {}
    start = 0
    for name, count in [
        ("left_leg", 6),
        ("right_leg", 6),
        ("waist", 3),
        ("left_arm", 7),
        ("left_hand", 7),
        ("right_arm", 7),
        ("right_hand", 7),
    ]:
        state[name] = {"start": start, "end": start + count}
        start += count
    state["projected_gravity"] = {"start": 0, "end": 3, "original_key": "observation.projected_gravity"}
    write_json(
        tmp_path / "meta/info.json",
        {"codebase_version": "v2.1", "fps": 50, "features": features, "total_frames": 2},
    )
    write_json(
        tmp_path / "meta/modality.json",
        {
            "action": action,
            "state": state,
            "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        },
    )
    write_json(
        tmp_path / "meta/sonic_provenance.json",
        {"controller_variant": "sonic_v1_1", "controller_sha256": "a" * 64},
    )
    (tmp_path / "data").mkdir()
    pq.write_table(
        pa.table(
            {
                "action": [[0.0] * 78] * 2,
                "observation.state": [[0.0] * 43] * 2,
                "observation.projected_gravity": [[0.0, 0.0, -1.0]] * 2,
            }
        ),
        tmp_path / "data/episode_000.parquet",
    )
    return tmp_path


def test_valid_g1_schema_checks_all_frames(g1_schema_fixture):
    assert validate_g1_dataset(g1_schema_fixture)["checked_frames"] == 2


@pytest.mark.parametrize("case", ["negative_token_slice", "overlapping_hands", "short_leg"])
def test_invalid_g1_joint_and_token_slices_are_rejected(g1_schema_fixture, case):
    path = g1_schema_fixture / "meta/modality.json"
    modality = read_json(path)
    if case == "negative_token_slice":
        modality["action"]["motion_token"].update(start=-1, end=63)
    elif case == "overlapping_hands":
        modality["action"]["right_hand_joints"].update(start=64, end=71)
    else:
        modality["state"]["left_leg"]["end"] = 5
    write_json(path, modality)
    with pytest.raises(ValueError, match="nonnegative|Overlapping|expected 6"):
        validate_g1_dataset(g1_schema_fixture)
