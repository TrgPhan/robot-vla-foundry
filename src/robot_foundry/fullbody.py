"""Simulation-only SONIC v1.1 G1 port of the pinned NVIDIA deployment observations.

Real encoder/decoder inference drives PD torques; the floating base is never reset
after initialization. Motion reference input is not a language-conditioned VLA.
"""

import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from .io import read_json, write_json

# policy_parameters.hpp: an Isaac index for each hardware/MuJoCo index and its inverse.
ISAAC_FOR_MJ = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
)
MJ_FOR_ISAAC = np.argsort(ISAAC_FOR_MJ)
DEFAULT_MJ = np.array(
    [
        -0.312,
        0,
        0,
        0.669,
        -0.363,
        0,
        -0.312,
        0,
        0,
        0.669,
        -0.363,
        0,
        0,
        0,
        0,
        0.2,
        0.2,
        0,
        0.6,
        0,
        0,
        0,
        0.2,
        -0.2,
        0,
        0.6,
        0,
        0,
        0,
    ]
)
JOINTS = (
    [
        f"{side}_{part}_joint"
        for side in ["left", "right"]
        for part in ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
    ]
    + [f"waist_{axis}_joint" for axis in ["yaw", "roll", "pitch"]]
    + [
        f"{side}_{part}_joint"
        for side in ["left", "right"]
        for part in [
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        ]
    ]
)
ENCODER_DIMS = {
    "encoder_mode_4": 4,
    "motion_joint_positions_10frame_step5": 290,
    "motion_joint_velocities_10frame_step5": 290,
    "motion_anchor_orientation_heading_10frame_step5": 60,
    "motion_anchor_orientation_heading": 6,
    "motion_joint_positions_lowerbody_10frame_step5": 120,
    "motion_joint_velocities_lowerbody_10frame_step5": 120,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "smpl_joints_10frame_step1": 720,
    "smpl_anchor_orientation_heading_10frame_step1": 60,
    "motion_joint_positions_wrists_10frame_step1": 60,
}
DECODER_DIMS = {
    "token_state": 64,
    "his_base_angular_velocity_10frame_step1": 30,
    "his_body_joint_positions_10frame_step1": 290,
    "his_body_joint_velocities_10frame_step1": 290,
    "his_last_actions_10frame_step1": 290,
    "his_gravity_dir_10frame_step1": 30,
}


def require_g1_vla_checkpoint(path):
    """Reject the actual Panda checkpoint instead of pretending its 7D actions are SONIC tokens."""
    manifest = read_json(Path(path) / "data_manifest.json")
    if manifest.get("action_convention") == "libero_osc_delta_xyz_axis_angle_gripper_open01":
        raise ValueError("Panda 7D OSC checkpoint cannot control G1. Train on paired G1/SONIC data first.")
    raise ValueError("This runtime supports SONIC reference tracking; use the official GR00T G1 VLA stack.")


def rotation_wxyz(quat):
    q = np.asarray(quat, np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Expected a finite nonzero WXYZ quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def heading_rotation(quat):
    direction = rotation_wxyz(quat)[:, 0]
    yaw = np.arctan2(direction[1], direction[0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def load_motion(root):
    arrays = {}
    for name in ["joint_pos", "joint_vel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel"]:
        arrays[name] = np.loadtxt(Path(root) / f"{name}.csv", delimiter=",", skiprows=1, ndmin=2)
    frames = len(arrays["joint_pos"])
    if frames < 2 or any(len(a) != frames or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Motion arrays must have matching finite frames")
    if any(arrays[n].shape[1] != 29 for n in ["joint_pos", "joint_vel"]):
        raise ValueError("The reference must contain 29 G1 joints in Isaac order")
    for name, width in [("body_pos", 3), ("body_quat", 4), ("body_lin_vel", 3), ("body_ang_vel", 3)]:
        if arrays[name].shape[1] % width:
            raise ValueError(f"Invalid reference dimensions: {name}")
        arrays[name] = arrays[name].reshape(frames, -1, width)
    return arrays


def motor_parameters():
    armature = np.array(
        [0.025101925, 0.025101925, 0.010177520, 0.025101925, 0.003609725, 0.003609725] * 2
        + [0.010177520, 0.003609725, 0.003609725]
        + [0.003609725] * 5
        + [0.00425] * 2
        + [0.003609725] * 5
        + [0.00425] * 2
    )
    effort = np.array(
        [139, 139, 88, 139, 25, 25] * 2 + [88, 25, 25] + [25] * 5 + [5] * 2 + [25] * 5 + [5] * 2
    )
    stiffness = armature * (20 * np.pi) ** 2
    kp, kd = stiffness.copy(), 4 * armature * 20 * np.pi
    kp[[4, 5, 10, 11, 13, 14]] *= 2
    kd[[4, 5, 10, 11, 13, 14]] *= 2
    # The official v1.1 launch applies 1.5 to ankle pitch, indices 4 and 10.
    kp[[4, 10]] *= 1.5
    kd[[4, 10]] *= 1.5
    return kp, kd, 0.25 * effort / stiffness


class SonicController:
    def __init__(self, models):
        import onnxruntime as ort
        import yaml

        config = yaml.safe_load(Path(models["observation_config.yaml"]).read_text(encoding="utf-8"))
        self.encoder_names = [x["name"] for x in config["encoder"]["encoder_observations"] if x["enabled"]]
        self.decoder_names = [x["name"] for x in config["observations"] if x["enabled"]]
        if self.encoder_names != list(ENCODER_DIMS) or self.decoder_names != list(DECODER_DIMS):
            raise ValueError("Observation layout differs from the pinned SONIC v1.1 contract")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(models["model_encoder.onnx"]), options, providers=["CPUExecutionProvider"]
        )
        self.decoder = ort.InferenceSession(
            str(models["model_decoder.onnx"]), options, providers=["CPUExecutionProvider"]
        )
        self._check_shape(self.encoder, 1751, 64)
        self._check_shape(self.decoder, 994, 29)
        self.history = deque(maxlen=10)
        for _ in range(9):
            self.history.append(
                {
                    "ang": np.zeros(3),
                    "q": np.zeros(29),
                    "dq": np.zeros(29),
                    "action": np.zeros(29),
                    "gravity": np.array([0.0, 0.0, -1.0]),
                }
            )
        self.last_action = np.zeros(29)

    @staticmethod
    def _check_shape(session, inputs, outputs):
        if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1:
            raise ValueError("Expected one observation input and one model output")
        if session.get_inputs()[0].shape[-1] != inputs or session.get_outputs()[0].shape[-1] != outputs:
            raise ValueError("SONIC ONNX dimensions do not match its observation config")

    @staticmethod
    def _infer(session, observation):
        obs = np.asarray(observation, np.float32)[None]
        if not np.isfinite(obs).all():
            raise ValueError("Nonfinite SONIC observation")
        result = session.run(None, {session.get_inputs()[0].name: obs})[0][0]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite SONIC prediction")
        return result

    def predict(self, q_mj, dq_mj, quat, ang_vel, motion, frame, playing=True):
        matrix = rotation_wxyz(quat)
        self.history.append(
            {
                "ang": ang_vel.copy(),
                "q": (q_mj - DEFAULT_MJ)[MJ_FOR_ISAAC],
                "dq": dq_mj[MJ_FOR_ISAAC],
                "action": self.last_action.copy(),
                "gravity": matrix.T @ np.array([0.0, 0.0, -1.0]),
            }
        )
        indices = np.minimum(frame + (np.arange(10) * 5 if playing else 0), len(motion["joint_pos"]) - 1)
        if np.ndim(indices) == 0:
            indices = np.full(10, indices)
        # C++ GatherEncoderMode stores scalar mode=0 in slot zero, NOT a one-hot vector.
        terms = {name: np.zeros(size) for name, size in ENCODER_DIMS.items()}
        terms["motion_joint_positions_10frame_step5"] = motion["joint_pos"][indices].ravel()
        terms["motion_joint_velocities_10frame_step5"] = (
            motion["joint_vel"][indices].ravel() if playing else np.zeros(290)
        )
        inverse_heading = heading_rotation(quat).T
        terms["motion_anchor_orientation_heading_10frame_step5"] = np.concatenate(
            [(inverse_heading @ rotation_wxyz(motion["body_quat"][i, 0]))[:, :2].ravel() for i in indices]
        )
        token = self._infer(self.encoder, np.concatenate([terms[n] for n in self.encoder_names]))
        histories = {}
        for name, key in zip(self.decoder_names[1:], ["ang", "q", "dq", "action", "gravity"], strict=True):
            histories[name] = np.concatenate([entry[key] for entry in self.history])
        histories["token_state"] = token
        action = self._infer(self.decoder, np.concatenate([histories[n] for n in self.decoder_names]))
        self.last_action = action.copy()
        return action, token


def evaluate_sonic(scene, motion_path, models, provenance, output, seconds=9.1, warmup=0.5):
    """Untethered free-base PD dynamics; render an external full-body camera to MP4."""
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    import av
    import mujoco
    from PIL import Image, ImageDraw

    if not np.isfinite([seconds, warmup]).all() or seconds < 0.02 or warmup < 0:
        raise ValueError("Durations must be finite; seconds positive and warmup nonnegative")
    output = Path(output)
    motion = load_motion(motion_path)
    if seconds > len(motion["joint_pos"]) / 50:
        raise ValueError("Requested duration exceeds the unique reference clip; do not silently loop")
    model = mujoco.MjModel.from_xml_path(str(scene))
    model.opt.timestep = 0.002
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    if model.nu != 29 or model.nq != 36 or model.nv != 35:
        raise ValueError("Expected a free-base 29-actuator G1 model")
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in model.actuator_trnid[:, 0]]
    if names != JOINTS:
        raise ValueError("MuJoCo actuator order does not match the checked SONIC hardware order")
    # The upstream scene has a static site named com_marker; it is not the robot's measured CoM.
    marker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "com_marker")
    if marker >= 0:
        model.site_rgba[marker, 3] = 0
    data = mujoco.MjData(model)
    data.qpos[:3] = motion["body_pos"][0, 0]
    data.qpos[3:7] = motion["body_quat"][0, 0]
    data.qpos[7:] = motion["joint_pos"][0, ISAAC_FOR_MJ]
    mujoco.mj_forward(model, data)
    controller = SonicController(models)
    kp, kd, action_scale = motor_parameters()
    output.mkdir(parents=True, exist_ok=False)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 3.2, 135, -15
    video_file = output / "fullbody.mp4"
    container = av.open(str(video_file), "w")
    stream = container.add_stream("libx264", rate=50)
    stream.width, stream.height, stream.pix_fmt = 960, 720, "yuv420p"
    stream.options = {"crf": "22", "preset": "fast"}
    samples, latencies, failed = [], [], None
    started = time.perf_counter()
    warmup_steps = round(warmup * 50)
    total_steps = warmup_steps + round(seconds * 50)
    renderer = None
    try:
        renderer = mujoco.Renderer(model, height=720, width=960)
        with (output / "telemetry.jsonl").open("w", encoding="utf-8") as log:
            for step in range(total_steps):
                playing = step >= warmup_steps
                frame = max(0, step - warmup_steps)
                before = time.perf_counter()
                action, token = controller.predict(
                    data.qpos[7:],
                    data.qvel[6:],
                    data.sensor("imu_quat").data.copy(),
                    data.sensor("imu_gyro").data.copy(),
                    motion,
                    frame,
                    playing,
                )
                latencies.append(time.perf_counter() - before)
                target = DEFAULT_MJ + action[ISAAC_FOR_MJ] * action_scale
                for _ in range(10):
                    torque = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
                    data.ctrl[:] = np.clip(
                        torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
                    )
                    mujoco.mj_step(model, data)
                matrix = rotation_wxyz(data.qpos[3:7])
                tilt = float(np.degrees(np.arccos(np.clip(matrix[2, 2], -1, 1))))
                sample = {
                    "step": step,
                    "time_s": float(data.time),
                    "phase": "tracking" if playing else "hold",
                    "reference_frame": frame,
                    "base_height_m": float(data.qpos[2]),
                    "tilt_deg": tilt,
                    "joint_rmse_rad": float(
                        np.sqrt(np.mean((data.qpos[7:] - motion["joint_pos"][frame, ISAAC_FOR_MJ]) ** 2))
                    ),
                    "base_xyz": data.qpos[:3].tolist(),
                    "q_mj": data.qpos[7:].tolist(),
                    "token_l2": float(np.linalg.norm(token)),
                    "action_l2": float(np.linalg.norm(action)),
                    "contact_count": int(data.ncon),
                }
                samples.append(sample)
                log.write(json.dumps(sample, allow_nan=False) + "\n")
                camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.65]
                renderer.update_scene(data, camera=camera)
                pixels = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(pixels)
                draw.rectangle((0, 0, 960, 48), fill=(12, 18, 25))
                draw.text(
                    (12, 8), "G1 FULL BODY | SONIC v1.1 pretrained | MuJoCo free-base dynamics", fill="white"
                )
                draw.text(
                    (12, 27),
                    f"Reference motion tracking (no VLA) | t={data.time:.2f}s | height={data.qpos[2]:.3f}m | tilt={tilt:.1f}deg",
                    fill="white",
                )
                image = av.VideoFrame.from_ndarray(np.asarray(pixels), format="rgb24")
                for packet in stream.encode(image):
                    container.mux(packet)
                if step % 50 == 0:
                    print(
                        f"SONIC step={step}/{total_steps} height={data.qpos[2]:.3f} tilt={tilt:.1f}",
                        flush=True,
                    )
                if data.qpos[2] < 0.35 or tilt > 75 or not np.isfinite(data.qpos).all():
                    failed = "Declared smoke fall criterion: pelvis below 0.35 m or tilt above 75 deg"
                    break
    finally:
        try:
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
            if renderer is not None:
                renderer.close()
    report = {
        "status": "complete",
        "robot": "unitree_g1_29dof",
        "controller": "sonic_v1_1",
        "input_mode": "official_g1_reference_encoder",
        "vla_used": False,
        "kaggle_panda_checkpoint_used": False,
        "scope": "full-body motion tracking smoke; no manipulation benchmark",
        "free_base": True,
        "elastic_band": False,
        "physics_hz": 500,
        "control_hz": 50,
        "video_fps": 50,
        "warmup_s": warmup,
        "requested_tracking_s": seconds,
        "completed_control_steps": len(samples),
        "completed_requested_duration": len(samples) == total_steps,
        "fall_criterion": {"min_pelvis_height_m": 0.35, "max_tilt_deg": 75},
        "fall_detected": failed is not None,
        "failure": failed,
        "min_base_height_m": min(s["base_height_m"] for s in samples),
        "max_tilt_deg": max(s["tilt_deg"] for s in samples),
        "mean_joint_rmse_rad": float(
            np.mean([s["joint_rmse_rad"] for s in samples if s["phase"] == "tracking"])
        )
        if any(s["phase"] == "tracking" for s in samples)
        else None,
        "onnx_providers": ["CPUExecutionProvider"],
        "inference_p50_ms": float(np.percentile(latencies, 50) * 1000),
        "inference_p95_ms": float(np.percentile(latencies, 95) * 1000),
        "elapsed_wall_s": time.perf_counter() - started,
        "video": video_file.name,
        "provenance": provenance,
    }
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}, indent=2), flush=True)
    return report
