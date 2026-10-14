"""MuJoCo G1 environment shared by synchronized teacher collection and learned VLA rollouts."""

import os

import numpy as np

from .fullbody import DEFAULT_MJ, ISAAC_FOR_MJ, JOINTS, MJ_FOR_ISAAC, motor_parameters, rotation_wxyz


def quat_product(a, b):
    a, b = np.asarray(a), np.asarray(b)
    w = a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1)
    xyz = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate([w[..., None], xyz], axis=-1)


def scenario_motion(motion, seed):
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-0.35, 0.35)
    shift = np.r_[rng.uniform(-0.025, 0.025, 2), 0.0]
    delta = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    rotation = rotation_wxyz(delta)
    result = {k: v.copy() for k, v in motion.items()}
    for key in ["body_pos", "body_lin_vel", "body_ang_vel"]:
        result[key] = result[key] @ rotation.T
    result["body_pos"] += shift
    result["body_quat"] = quat_product(delta, result["body_quat"])
    return result


class G1Environment:
    control_hz = 50
    physics_hz = 500

    def __init__(self, scene, motion, seed=42, image_size=256, video_size=None):
        if os.name != "nt":
            os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self.mj = mujoco
        self.image_size = image_size
        self.render_size = video_size or image_size
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = 1 / self.physics_hz
        self.model.vis.global_.offwidth = self.render_size
        self.model.vis.global_.offheight = self.render_size
        if (self.model.nu, self.model.nq, self.model.nv) != (29, 36, 35):
            raise ValueError("G1 latent policy requires the free-base 29-actuator robot")
        names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
            for j in self.model.actuator_trnid[:, 0]
        ]
        if names != JOINTS:
            raise ValueError("G1 actuator ordering mismatch")
        marker = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "com_marker")
        if marker >= 0:
            self.model.site_rgba[marker, 3] = 0
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:3] = motion["body_pos"][0, 0]
        self.data.qpos[3:7] = motion["body_quat"][0, 0]
        rng = np.random.default_rng(seed)
        self.data.qpos[7:] = motion["joint_pos"][0, ISAAC_FOR_MJ] + rng.uniform(-0.004, 0.004, 29)
        mujoco.mj_forward(self.model, self.data)
        self.kp, self.kd, self.action_scale = motor_parameters()
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = rng.uniform(3.0, 3.3)
        self.camera.azimuth = rng.uniform(110, 155)
        self.camera.elevation = rng.uniform(-18, -13)
        self.renderer = mujoco.Renderer(self.model, height=self.render_size, width=self.render_size)
        self.policy_renderer = (
            mujoco.Renderer(self.model, height=image_size, width=image_size)
            if self.render_size != image_size
            else None
        )

    def proprio(self):
        return (
            self.data.qpos[7:].copy(),
            self.data.qvel[6:].copy(),
            self.data.sensor("imu_quat").data.copy(),
            self.data.sensor("imu_gyro").data.copy(),
        )

    def state(self):
        q, dq, quat, ang = self.proprio()
        rotation = rotation_wxyz(quat)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        return np.concatenate(
            [
                q[MJ_FOR_ISAAC],
                dq[MJ_FOR_ISAAC],
                ang,
                rotation.T @ [0.0, 0.0, -1.0],
                rotation.T @ self.data.qvel[:3],
                [self.data.qpos[2], np.sin(yaw), np.cos(yaw)],
            ]
        ).astype(np.float32)

    def render(self):
        self.camera.lookat[:] = [self.data.qpos[0], self.data.qpos[1], 0.65]
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def image(self, pixels=None):
        from PIL import Image

        if pixels is None and self.policy_renderer is not None:
            self.camera.lookat[:] = [self.data.qpos[0], self.data.qpos[1], 0.65]
            self.policy_renderer.update_scene(self.data, camera=self.camera)
            return self.policy_renderer.render().copy()
        pixels = self.render() if pixels is None else pixels
        if pixels.shape[:2] != (self.image_size, self.image_size):
            pixels = np.asarray(Image.fromarray(pixels).resize((self.image_size, self.image_size)))
        return pixels

    def step(self, action):
        action = np.asarray(action)
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("Decoder output must be 29 finite Isaac-order actions")
        target = DEFAULT_MJ + action[ISAAC_FOR_MJ] * self.action_scale
        saturation = 0
        for _ in range(self.physics_hz // self.control_hz):
            torque = self.kp * (target - self.data.qpos[7:]) - self.kd * self.data.qvel[6:]
            clipped = np.clip(
                torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
            )
            saturation += int(np.count_nonzero(clipped != torque))
            self.data.ctrl[:] = clipped
            self.mj.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Nonfinite G1 dynamics")
        tilt = float(np.degrees(np.arccos(np.clip(rotation_wxyz(self.data.qpos[3:7])[2, 2], -1, 1))))
        height = float(self.data.qpos[2])
        return {
            "time_s": float(self.data.time),
            "height_m": height,
            "tilt_deg": tilt,
            "fall": height < 0.35 or tilt > 75,
            "saturated_torque_values": saturation,
        }

    def close(self):
        if self.policy_renderer is not None:
            self.policy_renderer.close()
        self.renderer.close()
