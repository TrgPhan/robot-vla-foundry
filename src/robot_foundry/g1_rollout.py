"""Evaluate learned G1 token policies with zero teacher-token fallback after initialization."""

import json
import time
from pathlib import Path

import numpy as np

from .fullbody import ISAAC_FOR_MJ, SonicController, load_motion, rotation_wxyz
from .g1_collect import CameraVideo
from .g1_env import G1Environment, scenario_motion
from .g1_schema import controller_identity, finite_token, load_g1_statistics, validate_g1_manifest
from .g1_transport import RemoteG1Policy
from .io import file_hash, read_json, write_json
from .sonic_assets import prepare_sonic_assets


def evaluate_g1(
    checkpoint,
    policy_url,
    output,
    trials_per_task=1,
    execution_horizon=4,
    max_frames=None,
    policy_mode="trained",
):
    from PIL import Image, ImageDraw

    checkpoint, output = Path(checkpoint), Path(output)
    manifest = read_json(checkpoint / "data_manifest.json")
    validate_g1_manifest(manifest)
    cfg = manifest["config"]
    if policy_mode not in ["trained", "mean_token"]:
        raise ValueError("Choose trained policy or explicit train-mean baseline")
    mean_token = np.asarray(load_g1_statistics(checkpoint, manifest)["action"]["mean"], np.float32)
    if not 1 <= execution_horizon <= cfg["horizon"] or trials_per_task < 1:
        raise ValueError("Invalid G1 evaluation horizon/trials")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    policy = RemoteG1Policy(policy_url, manifest)
    weight_file = checkpoint / ("model.pt" if policy.health["backend"] == "tiny_bc" else "model.safetensors")
    if file_hash(weight_file) != policy.health["weights_sha256"]:
        raise ValueError("Policy server has loaded different checkpoint weights")
    output.mkdir(parents=True, exist_ok=False)
    thresholds = {
        "mean_joint_rmse_rad": 0.25,
        "mean_root_orientation_error_rad": 0.35,
        "min_height_m": 0.35,
        "max_tilt_deg": 75.0,
    }
    trials, latencies = [], []
    start_time = time.perf_counter()
    for task_id, task in enumerate(cfg["tasks"]):
        cases = [r for r in manifest["episodes"] if r["task_id"] == task_id and r["split"] == "test"]
        if trials_per_task > len(cases):
            raise ValueError("Do not silently repeat test seeds; collect more held-out episodes")
        scene, path, models, provenance = prepare_sonic_assets(cfg["repo"], cfg["assets"], task["motion"])
        validate_g1_manifest(manifest, controller_identity(provenance))
        original = load_motion(path)
        for case in cases[:trials_per_task]:
            motion = scenario_motion(original, case["seed"])
            env = G1Environment(scene, motion, case["seed"], cfg["image_size"], video_size=512)
            controller = SonicController(models)
            relative_video = f"videos/task_{task_id:02d}_seed_{case['seed']}.mp4"
            video = CameraVideo(output / relative_video, 512)
            joint_errors, orientation_errors, metrics, tokens_log, requests = [], [], [], [], 0
            frames = min(case["length"], max_frames or case["length"])
            failure, chunk = None, None
            try:
                # Standard pose initialization only. Tracking below never calls the reference encoder.
                for _ in range(cfg["warmup_frames"]):
                    action, _ = controller.predict(*env.proprio(), motion, 0, playing=False)
                    if env.step(action)["fall"]:
                        raise RuntimeError("G1 fell during initialization before learned policy ran")
                teacher_count_at_start = getattr(controller, "reference_encoder_calls", 0)
                for frame in range(frames):
                    if frame % execution_horizon == 0:
                        if policy_mode == "trained":
                            before = time.perf_counter()
                            chunk = policy.predict(env.state(), env.image(), task["prompt"])
                            latencies.append(time.perf_counter() - before)
                            requests += 1
                        else:
                            chunk = np.tile(mean_token, (policy.horizon, 1))
                    token = finite_token(chunk[frame % execution_horizon])
                    # No motion/reference/time index is passed to this control interface.
                    action = controller.decode_token(*env.proprio(), token)
                    metric = env.step(action)
                    metrics.append(metric)
                    joint_error = float(
                        np.sqrt(np.mean((env.data.qpos[7:] - motion["joint_pos"][frame, ISAAC_FOR_MJ]) ** 2))
                    )
                    rotation_error = rotation_wxyz(env.data.qpos[3:7]).T @ rotation_wxyz(
                        motion["body_quat"][frame, 0]
                    )
                    orientation_error = float(np.arccos(np.clip((np.trace(rotation_error) - 1) / 2, -1, 1)))
                    joint_errors.append(joint_error)
                    orientation_errors.append(orientation_error)
                    tokens_log.append(
                        {
                            "frame": frame,
                            "token_source": "learned_policy"
                            if policy_mode == "trained"
                            else "train_mean_baseline",
                            "token": token.tolist(),
                            "joint_rmse_rad": joint_error,
                            "root_orientation_error_rad": orientation_error,
                            **metric,
                        }
                    )
                    image = Image.fromarray(env.render())
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 0, 512, 48), fill=(8, 18, 28))
                    draw.text(
                        (6, 6),
                        f"G1 {policy_mode} {policy.health['backend']} -> SONIC | body 29DOF",
                        fill="white",
                    )
                    draw.text(
                        (6, 26),
                        f"mode={policy_mode} | tilt={metric['tilt_deg']:.1f}",
                        fill="white",
                    )
                    video.write(np.asarray(image))
                    if metric["fall"]:
                        failure = "Declared height/tilt fall criterion"
                        break
            finally:
                try:
                    video.close()
                finally:
                    env.close()
            mean_joint, mean_orientation = float(np.mean(joint_errors)), float(np.mean(orientation_errors))
            complete = len(metrics) == frames
            teacher_tracking_calls = (
                getattr(controller, "reference_encoder_calls", 0) - teacher_count_at_start
            )
            if teacher_tracking_calls:
                raise RuntimeError(
                    "Reference encoder ran after initialization; rejecting a mislabeled learned rollout"
                )
            passed = (
                complete
                and failure is None
                and mean_joint <= thresholds["mean_joint_rmse_rad"]
                and mean_orientation <= thresholds["mean_root_orientation_error_rad"]
            )
            trial = {
                "task": task["motion"],
                "prompt": task["prompt"],
                "seed": case["seed"],
                "split": "test",
                "requested_frames": frames,
                "completed_frames": len(metrics),
                "survived": complete and failure is None,
                "tracking_pass": passed,
                "failure": failure,
                "mean_joint_rmse_rad": mean_joint,
                "mean_root_orientation_error_rad": mean_orientation,
                "max_tilt_deg": max(x["tilt_deg"] for x in metrics),
                "min_height_m": min(x["height_m"] for x in metrics),
                "policy_requests": requests,
                "learned_tokens_executed": len(metrics) if policy_mode == "trained" else 0,
                "teacher_calls_after_initialization": teacher_tracking_calls,
                "video": relative_video,
            }
            trials.append(trial)
            (output / f"tokens_task_{task_id}_seed_{case['seed']}.jsonl").write_text(
                "".join(json.dumps(x) + "\n" for x in tokens_log), encoding="utf-8"
            )
            with (output / "trials.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(trial) + "\n")
            print(
                f"LEARNED G1: {task['motion']} survived={trial['survived']} tracking_pass={passed}",
                flush=True,
            )
    report = {
        "status": "complete",
        "robot": manifest["signature"]["robot_type"],
        "policy_mode": policy_mode,
        "learned_policy_used": policy_mode == "trained",
        "vla_used": policy_mode == "trained" and policy.health["backend"] == "smolvla",
        "policy": policy.health,
        "controller_identity": manifest["controller_identity"],
        "scope": manifest["scope"],
        "benchmark_scope": manifest["benchmark_scope"],
        "full_body_actuators": 29,
        "hands_actuated": False,
        "free_base": True,
        "elastic_band": False,
        "teacher_calls_after_initialization": sum(t["teacher_calls_after_initialization"] for t in trials),
        "thresholds_declared_before_trials": thresholds,
        "execution_horizon": execution_horizon,
        "trials": len(trials),
        "survived_trials": sum(t["survived"] for t in trials),
        "tracking_passes": sum(t["tracking_pass"] for t in trials),
        "per_trial": trials,
        "policy_rpc_latency_p50_ms": float(np.percentile(latencies, 50) * 1000) if latencies else None,
        "policy_rpc_latency_p95_ms": float(np.percentile(latencies, 95) * 1000) if latencies else None,
        "wall_s": time.perf_counter() - start_time,
        "note": "Seen motion primitives, held-out rollout seeds; not object manipulation or sim-to-real",
    }
    write_json(output / "report.json", report)
    return report
