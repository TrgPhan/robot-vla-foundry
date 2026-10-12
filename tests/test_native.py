import ast
import os
from pathlib import Path

import pytest

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
