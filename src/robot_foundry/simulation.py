"""Closed-loop LIBERO rollouts with explicit observation and action conventions."""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from .backends import SmolBackend
from .data import tokenize
from .io import file_hash, read_json, write_json
from .model import TinyChunkPolicy
from .precision import autocast
from .train import seed_everything, to_device


def quat_xyzw_to_axis_angle(quat):
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    den = np.linalg.norm(q[:3])
    return (
        np.zeros(3, np.float32)
        if den < 1e-8
        else (q[:3] / den * (2 * math.atan2(den, q[3]))).astype(np.float32)
    )


def decode_libero_action(action):
    a = np.array(action, dtype=np.float64, copy=True)
    if a.shape != (7,) or not np.isfinite(a).all():
        raise ValueError("LIBERO requires a finite 7D OSC action.")
    a[:6] = np.clip(a[:6], -1, 1)
    # IPEC conversion uses 0=closed, 1=open; LIBERO OSC uses +1=closed, -1=open.
    a[6] = -1.0 if a[6] >= 0.5 else 1.0
    return a


class CheckpointPredictor:
    def __init__(self, checkpoint, backend, device="auto", precision="auto"):
        self.root = Path(checkpoint).resolve()
        self.manifest = read_json(self.root / "data_manifest.json")
        if self.manifest["action_convention"] != "libero_osc_delta_xyz_axis_angle_gripper_open01":
            raise ValueError("This simulation adapter only supports the explicit LIBERO action convention.")
        stats = read_json(self.root / "stats.json")
        self.stats = {
            key: {k: np.asarray(v, np.float32) for k, v in values.items()} for key, values in stats.items()
        }
        self.device = to_device({"device": device})
        self.backend_name = backend
        self.precision = (
            ("fp16" if backend == "smolvla" and self.device.type == "cuda" else "fp32")
            if precision == "auto"
            else precision
        )
        self.horizon = self.manifest["config"]["horizon"]
        self.image_size = self.manifest["config"]["image_size"]
        if backend == "tiny_bc":
            ckpt = torch.load(self.root / "model.pt", map_location=self.device, weights_only=True)
            self.model = TinyChunkPolicy(**ckpt["spec"]).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
        else:
            cfg = {
                "output_dir": str(self.root.parent),
                "registry": "configs/datasets.json",
                "precision": self.precision,
            }
            dataset = SimpleNamespace(root=self.root, manifest=self.manifest)
            self.backend = SmolBackend(cfg, dataset, self.device, load_path=self.root)
            self.backend.model.eval()

    @torch.inference_mode()
    def predict(self, obs, prompt):
        state = np.concatenate(
            [
                obs["robot0_eef_pos"],
                quat_xyzw_to_axis_angle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]
        ).astype(np.float32)
        image_map = {
            "observation.images.image": obs["agentview_image"],
            "observation.images.wrist_image": obs["robot0_eye_in_hand_image"],
        }
        images = []
        for camera in self.manifest["signature"]["cameras"]:
            # Matches the reference NVIDIA LIBERO wrapper's 180 degree image rotation.
            pixels = image_map[camera][::-1, ::-1].copy()
            pixels = np.asarray(
                Image.fromarray(pixels).resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            )
            images.append(pixels.transpose(2, 0, 1).copy())
        normalized_state = (state - self.stats["observation.state"]["mean"]) / self.stats[
            "observation.state"
        ]["std"]
        raw = {
            "images": torch.from_numpy(np.stack(images)).unsqueeze(0),
            "state": torch.from_numpy(
                normalized_state if self.backend_name == "tiny_bc" else state
            ).unsqueeze(0),
            "tokens": torch.from_numpy(tokenize(prompt)).unsqueeze(0),
            "prompt": [prompt],
            "action": torch.zeros(1, self.horizon, 7),
            "valid": torch.ones(1, self.horizon, dtype=torch.bool),
        }
        if self.backend_name == "tiny_bc":
            pred = self.model(
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
            )
        else:
            with autocast(self.device, self.precision):
                pred = self.backend.predict(self.backend.process(raw))
        pred = pred[0].float().cpu().numpy()
        return pred * self.stats["action"]["std"] + self.stats["action"]["mean"]


def wilson(successes, trials):
    if trials < 1:
        raise ValueError("Need at least one trial.")
    z = 1.959963984540054
    p = successes / trials
    den = 1 + z**2 / trials
    middle = (p + z**2 / (2 * trials)) / den
    radius = z * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / den
    return [max(0.0, middle - radius), min(1.0, middle + radius)]


def evaluate_libero(args):
    if args.episodes_per_task < 1 or args.max_steps < 1 or args.execution_horizon < 1:
        raise ValueError("Positive episode count, max steps, and execution horizon required.")
    # Set before importing MuJoCo/OpenGL. Windows uses its default WGL backend.
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # Use a project-owned config instead of LIBERO's interactive import prompt or the user's ~/.libero.
    import importlib.util

    spec = importlib.util.find_spec("libero")
    if spec is None:
        raise ImportError("Install the isolated LIBERO simulation environment using scripts/setup_sim.py.")
    package_root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    config_dir = Path(os.environ.setdefault("LIBERO_CONFIG_PATH", str(Path("data/sim_config").resolve())))
    if not (config_dir / "config.yaml").exists():
        config_dir.mkdir(parents=True, exist_ok=True)
        dataset_dir = Path("data/libero_sim_datasets").resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "benchmark_root": str(package_root),
            "bddl_files": str(package_root / "bddl_files"),
            "init_states": str(package_root / "init_files"),
            "assets": str(package_root / "assets"),
            "datasets": str(dataset_dir),
        }
        # JSON is a YAML subset accepted by LIBERO's yaml.load.
        (config_dir / "config.yaml").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    seed_everything(args.seed)
    if getattr(args, "policy_url", None):
        from .transport import RemotePredictor

        predictor = RemotePredictor(args.policy_url, read_json(Path(args.checkpoint) / "data_manifest.json"))
    else:
        predictor = CheckpointPredictor(args.checkpoint, args.backend, args.device)
    if args.execution_horizon > predictor.horizon:
        raise ValueError("Execution horizon exceeds model action chunk.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trials_file = output / "trials.jsonl"
    if trials_file.exists():
        raise FileExistsError("Use a fresh simulation output directory to avoid mixing trials.")
    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    trials = []
    task_metrics = []
    for task_id in range(task_suite.get_num_tasks()):
        task = task_suite.get_task(task_id)
        # These pinned LIBERO assets contain NumPy arrays serialized with pickle; torch>=2.6
        # changed its default. Load the official benchmark asset explicitly, never arbitrary input.
        init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        init_states = torch.load(init_path, map_location="cpu", weights_only=False)
        if args.episodes_per_task > len(init_states):
            raise ValueError(
                "Requested more trials than unique official initial states; do not silently repeat them."
            )
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256,
            camera_widths=256,
        )
        success_count = 0
        try:
            for episode in range(args.episodes_per_task):
                env.seed(args.seed + task_id * 1000 + episode)
                env.reset()
                obs = env.set_init_state(init_states[episode])
                for _ in range(10):
                    obs, _, _, _ = env.step(np.zeros(7))
                success = False
                chunk = None
                steps = 0
                for step in range(args.max_steps):
                    if step % args.execution_horizon == 0:
                        chunk = predictor.predict(obs, task.language)
                    obs, _, done, _ = env.step(decode_libero_action(chunk[step % args.execution_horizon]))
                    steps = step + 1
                    success = bool(env.check_success())
                    if success or done:
                        break
                success_count += int(success)
                trial = {
                    "suite": args.suite,
                    "task_id": task_id,
                    "task": task.name,
                    "initial_state_id": episode,
                    "seed": args.seed + task_id * 1000 + episode,
                    "success": success,
                    "steps": steps,
                }
                trials.append(trial)
                with trials_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(trial) + "\n")
                print(f"{task.name}: trial={episode + 1}, success={success}", flush=True)
        finally:
            env.close()
        task_metrics.append(
            {
                "task": task.name,
                "successes": success_count,
                "trials": args.episodes_per_task,
                "rate": success_count / args.episodes_per_task,
                "wilson_95": wilson(success_count, args.episodes_per_task),
            }
        )
    total_success = sum(r["success"] for r in trials)
    model_file = Path(args.checkpoint) / ("model.pt" if args.backend == "tiny_bc" else "model.safetensors")
    report = {
        "suite": args.suite,
        "backend": args.backend,
        "checkpoint_sha256": file_hash(model_file),
        "successes": total_success,
        "trials": len(trials),
        "macro_success_rate": float(np.mean([t["rate"] for t in task_metrics])),
        "wilson_95_descriptive_pooled": wilson(total_success, len(trials)),
        "tasks": task_metrics,
        "max_steps": args.max_steps,
        "execution_horizon": args.execution_horizon,
        "seed": args.seed,
        "note": "Pooled interval is descriptive; tasks are heterogeneous. No humanoid balance measured.",
    }
    write_json(output / "report.json", report)
    return report
