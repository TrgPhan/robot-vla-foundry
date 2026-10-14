"""Materialize pinned SONIC ONNX, MuJoCo meshes and one official motion clip."""

import hashlib
import os
import re
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .io import file_hash, read_json, write_json
from .upstream import check_checkout


def prepare_sonic_assets(repo, destination, motion_name):
    from huggingface_hub import hf_hub_download

    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", motion_name):
        raise ValueError("Use the name of one official reference motion directory")
    lock = check_checkout("sonic", repo)
    model = read_json("configs/datasets.json")["sonic"]
    destination.mkdir(parents=True, exist_ok=True)
    cached_models = [
        destination / "policy/sonic_v1_1" / name
        for name in ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]
    ]
    cached_motion = destination / "gear_sonic_deploy/reference/example" / motion_name
    robot_xml = destination / "gear_sonic_deploy/g1/g1_29dof_old.xml"
    geometry_cached = False
    if robot_xml.is_file():
        try:
            geometry_cached = all(
                (robot_xml.parent / "meshes" / x.attrib["file"]).is_file()
                for x in ET.parse(robot_xml).iter("mesh")
            )
        except ET.ParseError:
            pass
    reusable = (
        geometry_cached
        and all(p.is_file() and p.stat().st_size > 0 for p in cached_models)
        and all(
            (cached_motion / name).is_file()
            for name in [
                "joint_pos.csv",
                "joint_vel.csv",
                "body_pos.csv",
                "body_quat.csv",
                "body_lin_vel.csv",
                "body_ang_vel.csv",
            ]
        )
    )
    reserve = 128 * 2**20 if reusable else 2 * 2**30
    if shutil.disk_usage(destination).free < reserve:
        raise OSError(28, f"SONIC needs {reserve / 2**30:.2f} GiB free for assets/rollout outputs")

    def copy_upstream(relative):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Expected a relative committed asset path")
        # Read the committed bytes, so local modifications cannot silently change the robot/reference.
        content = subprocess.run(
            ["git", "-C", str(repo), "show", f"{lock['sha']}:{relative}"],
            capture_output=True,
            check=True,
        ).stdout
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved = target.resolve()
        if not resolved.is_relative_to(destination):
            raise ValueError(f"Upstream asset path must stay inside destination: {resolved}")
        pointer = content.startswith(b"version https://git-lfs.github.com/spec/v1")
        expected = re.search(rb"oid sha256:([a-f0-9]{64})", content) if pointer else None
        digest = expected[1].decode() if expected else hashlib.sha256(content).hexdigest()
        if target.exists() and file_hash(target) == digest:
            return target
        temp = target.with_suffix(target.suffix + ".part")
        try:
            if pointer:
                url = f"https://media.githubusercontent.com/media/{lock['repo']}/{lock['sha']}/{relative}"
                with urllib.request.urlopen(url, timeout=60) as response, temp.open("wb") as out:
                    shutil.copyfileobj(response, out, 2**20)
            else:
                temp.write_bytes(content)
            if file_hash(temp) != digest:
                raise ValueError(f"Upstream asset hash mismatch: {relative}")
            os.replace(temp, target)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        print(f"Verified SONIC asset: {relative}", flush=True)
        return target

    g1 = "gear_sonic_deploy/g1"
    scene = copy_upstream(f"{g1}/scene_29dof.xml")
    robot = copy_upstream(f"{g1}/g1_29dof_old.xml")
    meshes = [f"{g1}/meshes/{mesh.attrib['file']}" for mesh in ET.parse(robot).iter("mesh")]
    motion = f"gear_sonic_deploy/reference/example/{motion_name}"
    references = [
        f"{motion}/{name}"
        for name in [
            "metadata.txt",
            "joint_pos.csv",
            "joint_vel.csv",
            "body_pos.csv",
            "body_quat.csv",
            "body_lin_vel.csv",
            "body_ang_vel.csv",
        ]
    ]
    with ThreadPoolExecutor(max_workers=4) as workers:
        paths = list(workers.map(copy_upstream, meshes + references))
    models = {}
    for name in ["model_encoder.onnx", "model_decoder.onnx", "observation_config.yaml"]:
        models[name] = Path(
            hf_hub_download(
                model["repo_id"],
                f"sonic_v1_1/{name}",
                revision=model["revision"],
                local_dir=destination / "policy",
            )
        )
    provenance = {
        "robot": "unitree_g1_29dof",
        "controller_variant": "sonic_v1_1",
        "upstream": lock,
        "model": model,
        "motion": motion_name,
        "files": {
            p.relative_to(destination).as_posix(): {"sha256": file_hash(p), "bytes": p.stat().st_size}
            for p in [scene, robot, *paths, *models.values()]
        },
    }
    write_json(destination / "provenance.json", provenance)
    return scene, destination / motion, models, provenance
