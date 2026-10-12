import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def launcher():
    spec = importlib.util.spec_from_file_location("kaggle_launcher", Path("scripts/run_kaggle.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(**overrides):
    values = dict(
        gpus="auto",
        backend="smolvla",
        name="test",
        mode="full",
        full_data=False,
        microbatch=None,
        precision="auto",
        workers=2,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def hardware(count):
    return {"gpus": [{"total_gib": 14.75, "kernel_error": None} for _ in range(count)]}


def test_t4x2_keeps_effective_batch_and_uses_fp16():
    runner = launcher()
    one = runner.recipe(arguments(gpus="1"), hardware(2))
    two = runner.recipe(arguments(gpus="2"), hardware(2))
    bigger_micro = runner.recipe(arguments(gpus="2", microbatch=2), hardware(2))
    for cfg in [one, two, bigger_micro]:
        assert cfg["batch_size"] * cfg["gradient_accumulation"] * cfg["num_gpus"] == 16
        assert cfg["precision"] == "fp16" and cfg["packed_cache"]
    assert two["num_gpus"] == 2 and two["gradient_accumulation"] == 8
    assert bigger_micro["gradient_accumulation"] == 4


def test_explicit_two_gpus_does_not_silently_fall_back():
    with pytest.raises(RuntimeError, match="detected 1"):
        launcher().recipe(arguments(gpus="2"), hardware(1))


def test_benchmark_compares_the_same_effective_batch():
    runner = launcher()
    for count in [1, 2]:
        cfg = runner.recipe(arguments(gpus=str(count), mode="benchmark"), hardware(2))
        assert cfg["batch_size"] * cfg["gradient_accumulation"] * count == 16
        assert cfg["steps"] == 100
