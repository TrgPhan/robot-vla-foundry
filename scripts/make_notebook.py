"""Generate G1 collection -> real VLA training -> trained full-body inference notebooks."""

import json
import subprocess
import sys
from pathlib import Path


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell(
        "markdown",
        """# Train a G1 full-body VLA on Kaggle, then infer the saved checkpoint

Enable Internet + GPU T4 x2. This notebook includes **actual optimizer training**, not only SONIC playback.
Public G1 motion references -> paired simulation RGB/state/64D tokens -> pretrained SmolVLA training
-> saved trained VLA predicts tokens -> frozen SONIC decoder -> 29 G1 body actuators -> full-body MP4.

Scope: language-conditioned locomotion/gestures, external camera, fixed hand meshes.
Object grasping/Dex3 fingers need a different paired dataset/scene. The Panda notebook is kaggle_panda_train.ipynb.
Local integration used real SmolVLA G1 optimizer updates and checkpoint reload/rollout;
5,000-step T4 x2 convergence still needs evaluation on your run. Start with SMOKE=True.
""",
    ),
    cell(
        "code",
        """from pathlib import Path
import os, sys, json, subprocess, shutil
REPO_URL = 'https://github.com/TrgPhan/robot-vla-foundry.git'
GIT_REF = 'g1-vla-v1'
PROJECT = Path('/kaggle/working/robot-vla-foundry')
def run(argv, **kwargs):
    subprocess.run([str(x) for x in argv], check=True, **kwargs)
if not (PROJECT / '.git').exists():
    if PROJECT.exists() and any(PROJECT.iterdir()):
        raise FileExistsError('Use an empty project directory.')
    run(['git','clone','--depth','1','--branch',GIT_REF,REPO_URL,PROJECT])
else:
    origin = subprocess.check_output(['git','-C',str(PROJECT),'remote','get-url','origin'],text=True).strip()
    if origin != REPO_URL:
        raise ValueError('Existing checkout has a different remote.')
    if subprocess.check_output(['git','-C',str(PROJECT),'status','--porcelain'],text=True).strip():
        raise ValueError('Preserve source edits and use a fresh project directory.')
    run(['git','fetch','--depth','1','origin','tag',GIT_REF],cwd=PROJECT)
    run(['git','checkout','--detach',GIT_REF],cwd=PROJECT)
os.chdir(PROJECT)
print('Source:', subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
""",
    ),
    cell(
        "code",
        """run([sys.executable, 'scripts/setup_kaggle.py'])
TRAIN_PY = Path('/kaggle/working/robot-vla-env/bin/python')
run([TRAIN_PY, '-m', 'robot_foundry.cli', 'doctor'])
run([TRAIN_PY, 'scripts/setup_fullbody_sim.py'])
SIM_PY = PROJECT / '.venv-fullbody/bin/python'
print('Free disk GiB:', shutil.disk_usage(PROJECT).free / 2**30)
""",
    ),
    cell(
        "markdown",
        """## 1. Generate synchronized demonstrations (data preparation)

Capture state/camera before each target action. The teacher encoder supplies labels;
the real SONIC decoder and PD/contact dynamics execute them. Stats use only training episodes.
Splits hold out complete rollout seeds/camera/initial poses but share source motion primitives:
this is scenario interpolation, not unseen-motion generalization. Failed teacher rollouts stop collection.
Prepared RGB is lossless PNG; raw LeRobot v2.1 export videos use H264.
""",
    ),
    cell(
        "code",
        """DATASET_NAME = 'g1_motion_v1'
GENERATE_DATA = True  # False only if a compatible prepared manifest already exists.
data_config = json.loads(Path('configs/g1_data.json').read_text())
data_config['prepared_dir'] = f'data/prepared/{DATASET_NAME}'
data_config['raw_dir'] = f'data/g1_generated/{DATASET_NAME}'
# Default per motion task: 8 train + 2 validation + 2 test episodes, 50 Hz.
DATA_CONFIG = Path('outputs', DATASET_NAME + '_collection.json')
DATA_CONFIG.parent.mkdir(exist_ok=True)
DATA_CONFIG.write_text(json.dumps(data_config, indent=2))
if GENERATE_DATA:
    run([SIM_PY, 'scripts/collect_g1_data.py', '--config', DATA_CONFIG])
manifest = json.loads(Path(data_config['prepared_dir'], 'manifest.json').read_text())
assert manifest['action_convention'] == 'g1_sonic_v1_1_latent64'
assert manifest['signature']['state']['shape'] == [70]
assert manifest['signature']['action']['shape'] == [64]
print('DATA READY:', manifest['split_counts'], manifest['scope'])
""",
    ),
    cell(
        "markdown",
        """## 2. Train SmolVLA (optimizer updates)

Keep the pretrained VLM/action expert; initialize new G1 projections for state70 (padded96)/action64.
Freeze VLM, train the state/action projections and action expert. SONIC stays frozen as the whole-body prior/controller.
T4 x2 uses FP16/GradScaler, DDP and batch16 = 2 GPUs x microbatch1 x accumulation8.
Keep at least 6 GiB disk free for model/optimizer exports. DDP does not pool the GPUs' VRAM.
SMOKE=True runs 20 updates; False uses the 5,000-step recipe. Use a new run name when changing options.
""",
    ),
    cell(
        "code",
        """BACKEND = 'smolvla'  # tiny_bc is a learned baseline, not a foundation VLA.
SMOKE = True
RESUME = False
GPUS = 'auto'
MICROBATCH = 1  # Try 2 only after a fresh memory-measured smoke run.
WORKERS = 2
MODE = 'smoke' if SMOKE else 'full'
RUN_NAME = f'g1_{BACKEND}_{MODE}_mb{MICROBATCH}'
training = [TRAIN_PY, 'scripts/run_g1_train.py', '--name', RUN_NAME,
            '--prepared', data_config['prepared_dir'], '--backend', BACKEND,
            '--mode', MODE, '--gpus', GPUS, '--microbatch', MICROBATCH, '--workers', WORKERS]
if RESUME:
    training += ['--resume']
run(training)
report = json.loads(Path('outputs', RUN_NAME, 'report.json').read_text())
assert report['optimizer_updates'] > 0 and report['action_dimension'] == 64
print('TRAIN COMPLETE:', report['optimizer_updates'], '/', report['steps'], 'AMP skips:', report['amp_skipped_updates'])
print('Held-out latent RMSE:', report['test']['normalized_rmse'], '(not closed-loop success)')
""",
    ),
    cell(
        "markdown",
        """## 3. Infer the trained policy on the full G1 body

Load outputs/RUN_NAME/best, never an arm checkpoint or untrained base fallback.
After pose initialization, predicted tokens alone drive SONIC; zero reference-token fallback calls.
Video/report identify trained weights, optimizer updates, controller identity and token source.
Declared joint/orientation thresholds measure internal motion tracking, not object manipulation success.
Compare with the train-mean token baseline: passing a short smoke does not prove task learning.
""",
    ),
    cell(
        "code",
        """RUN_TRAINED_SIM = True
RUN_MEAN_BASELINE = True
N_TRIALS = 2  # Two unique test seeds/task in the default dataset; do not repeat seeds silently.
def simulate(argv):
    result = subprocess.run([str(x) for x in argv], check=False)
    if result.returncode:
        print('Inference failed; inspect the corresponding g1_simulation_status.json or g1_mean_baseline_status.json under outputs/RUN_NAME. Trained artifacts are retained.')
    return result.returncode
if RUN_TRAINED_SIM:
    returncode = simulate([TRAIN_PY, 'scripts/run_g1_vla_sim.py', '--run', RUN_NAME, '--skip-setup',
         '--trials', N_TRIALS, '--output', f'outputs/g1_eval_{RUN_NAME}'])
    if returncode == 0:
        result = json.loads(Path('outputs', f'g1_eval_{RUN_NAME}', 'report.json').read_text())
        print('TRAINED FULL-BODY:', result['tracking_passes'], '/', result['trials'])
        print('Teacher calls after initialization:', result['teacher_calls_after_initialization'])
if RUN_MEAN_BASELINE:
    returncode = simulate([TRAIN_PY, 'scripts/run_g1_vla_sim.py', '--run', RUN_NAME, '--skip-setup',
         '--trials', N_TRIALS, '--policy-mode', 'mean_token', '--output', f'outputs/g1_mean_{RUN_NAME}'])
    if returncode == 0:
        baseline = json.loads(Path('outputs', f'g1_mean_{RUN_NAME}', 'report.json').read_text())
        print('MEAN TOKEN BASELINE:', baseline['tracking_passes'], '/', baseline['trials'])
""",
    ),
    cell(
        "code",
        """from IPython.display import Video, display
VIDEO_DIR = Path('outputs', f'g1_eval_{RUN_NAME}', 'videos')
if VIDEO_DIR.exists():
    for video in sorted(VIDEO_DIR.glob('*.mp4')):
        print(video.name)
        display(Video(str(video), embed=True))
print('Retain trained artifacts:', PROJECT / 'outputs' / RUN_NAME)
print('Retain prepared data for resume:', PROJECT / data_config['prepared_dir'])
print('Retain evaluation reports/videos:', PROJECT / 'outputs' / f'g1_eval_{RUN_NAME}')
print('Free disk GiB:', shutil.disk_usage(PROJECT).free / 2**30)
""",
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, c in enumerate(cells):
    c["id"] = f"g1-foundry-{i:02d}"
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"g1_cell_{i}", "exec")
Path("notebooks").mkdir(exist_ok=True)
payload = json.dumps(notebook, indent=2) + "\n"
for name in ["kaggle_train.ipynb", "kaggle_g1_train.ipynb"]:
    Path("notebooks", name).write_text(payload, encoding="utf-8")
subprocess.run([sys.executable, "scripts/make_panda_notebook.py"], check=True)
print("Wrote syntax-checked G1 training+inference notebooks; retained the Panda alternative")
