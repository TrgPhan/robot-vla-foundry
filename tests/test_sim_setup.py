import importlib.util
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_foundry.io import read_json, write_json


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_run(project):
    out = project / "outputs" / "trained"
    write_json(out / "run_config.json", {"backend": "smolvla", "precision": "fp16"})
    write_json(out / "report.json", {"steps": 5000})
    write_json(out / "best/data_manifest.json", {"fingerprint": "test"})
    (out / "best/model.safetensors").write_bytes(b"saved-policy-must-remain")
    (out / "training_state.pt").write_bytes(b"saved-optimizer-must-remain")
    return out


def test_simulation_disk_guard_precedes_installer_work(tmp_path, monkeypatch):
    setup = load_script("setup_sim")
    monkeypatch.setattr(setup.shutil, "disk_usage", lambda _: SimpleNamespace(free=2 * 2**30))
    monkeypatch.setattr(setup.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(setup.Path, "cwd", lambda: tmp_path)
    monkeypatch.setattr(setup.sys, "argv", ["setup_sim.py", "--env", str(tmp_path / ".venv-sim")])
    invoked = []
    monkeypatch.setattr(setup, "run", invoked.append)
    with pytest.raises(OSError, match="Simulation installation needs 4 GiB"):
        setup.main()
    assert invoked == []
    assert not (tmp_path / ".venv-sim").exists()


@pytest.mark.parametrize("fail_stage", ["setup", "policy_server", "rollout"])
def test_optional_failure_preserves_training_and_stops_owned_server(tmp_path, monkeypatch, fail_stage):
    runner = load_script("run_kaggle_sim")
    out = saved_run(tmp_path)
    original_report = (out / "report.json").read_bytes()
    sim_python = (
        tmp_path / ".venv-sim-cpu" / ("Scripts/python.exe" if runner.os.name == "nt" else "bin/python")
    )
    sim_python.parent.mkdir(parents=True)
    sim_python.touch()
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        stage = (
            "setup"
            if "scripts/setup_sim.py" in argv
            else "policy_server"
            if "scripts/wait_policy_server.py" in argv
            else "rollout"
        )
        if stage == fail_stage:
            raise subprocess.CalledProcessError(2, argv)

    server = SimpleNamespace(poll=lambda: None, terminate=lambda: None, wait=lambda **_: None)
    stopped = []
    monkeypatch.setattr(runner.subprocess, "run", command)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: server)
    monkeypatch.setattr(runner.socket, "socket", lambda: nullcontext(SimpleNamespace(bind=lambda _: None)))
    monkeypatch.setattr(runner, "stop_server", stopped.append)
    report = runner.evaluate(tmp_path, "trained")
    assert report["status"] == "failed" and report["stage"] == fail_stage
    assert read_json(out / "simulation_status.json") == report
    assert (out / "report.json").read_bytes() == original_report
    assert (out / "best/model.safetensors").read_bytes() == b"saved-policy-must-remain"
    assert (out / "training_state.pt").read_bytes() == b"saved-optimizer-must-remain"
    assert not any("train" in argv for argv in calls)
    assert stopped == [None if fail_stage == "setup" else server]
