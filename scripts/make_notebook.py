"""Generate a syntax-checked Kaggle notebook that clones the public GitHub release."""

import json
from pathlib import Path


def cell(kind, text):
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell(
        "markdown",
        """# Robot VLA Foundry — clone GitHub, train on Kaggle

Repository: https://github.com/TrgPhan/robot-vla-foundry

Enable **Internet** and a **GPU** in Kaggle Settings. No Dataset upload or GitHub token is needed.
**Scope: Panda arm manipulation only. This notebook does not train G1/full-body control.**
For SONIC G1 reference simulation, see docs/FULLBODY.md.
The default is pretrained SmolVLA with automatic 1/2-GPU detection; select **GPU T4 x2** for DDP.
Each GPU needs >=14 GiB VRAM. GR00T has a separate >=40 GB Linux recipe.
Run the 20-step smoke experiment first; then use a fresh name for full training.
This notebook is syntax-checked and its policy/data components are tested locally;
full Kaggle training depends on the allocated hardware and network.
""",
    ),
    cell(
        "code",
        """from pathlib import Path
import os, sys, subprocess, shutil, json

REPO_URL = 'https://github.com/TrgPhan/robot-vla-foundry.git'
GIT_REF = 'kaggle-v2.5'  # Includes checkpoint download, local inference and optional rollout videos.
PROJECT = Path('/kaggle/working/robot-vla-foundry')

def run(argv, **kwargs):
    subprocess.run([str(x) for x in argv], check=True, **kwargs)

if not (PROJECT / '.git').exists():
    if PROJECT.exists() and any(PROJECT.iterdir()):
        raise FileExistsError('PROJECT is nonempty and not a Git checkout; select a fresh directory.')
    run(['git', 'clone', '--depth', '1', '--branch', GIT_REF, REPO_URL, PROJECT])
else:
    origin = subprocess.check_output(['git', '-C', str(PROJECT), 'remote', 'get-url', 'origin'], text=True).strip()
    if origin != REPO_URL:
        raise ValueError('Existing checkout has a different remote; select a fresh directory.')
    wanted = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', GIT_REF + '^{commit}'], text=True).strip()
    actual = subprocess.check_output(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != wanted:
        raise ValueError('Existing checkout has a different revision; select a fresh directory.')
os.chdir(PROJECT)
print('Project:', PROJECT)
print('Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
""",
    ),
    cell(
        "code",
        """run([sys.executable, 'scripts/setup_kaggle.py'])
TRAIN_PY = Path('/kaggle/working/robot-vla-env/bin/python')
run([TRAIN_PY, '-m', 'robot_foundry.cli', 'doctor'])
""",
    ),
    cell(
        "markdown",
        """## Select the experiment

SMOKE=True runs 20 steps. SMOKE=False uses the full 5,000-step recipe.
FULL_DATA=True selects all public Object/Spatial episodes; the default caps at 10 episodes/task.
Keep GPU count, precision, microbatch, workers and other options unchanged when resuming.
T4/P100 use FP16 autocast + GradScaler. CUDA 11.8 wheels retain older GPU support.
Two T4s use one process/GPU, disjoint samples and gradient synchronization after accumulation.
Full training keeps effective batch 16: two GPUs x microbatch 1 x accumulation 8.
Use `scripts/run_kaggle.py --mode benchmark --gpus 1` and `--gpus 2` in separate runs
to compare the same effective batch; metrics.jsonl records throughput, per-GPU VRAM and AMP skips.
Prepared NPZ duplicates are verified against packed RGB/labels and removed episode by episode.
Training requires at least 6 GiB free disk after staging SmolVLA; disk and VRAM are separate budgets.
""",
    ),
    cell(
        "code",
        """BACKEND = 'smolvla'  # or 'tiny_bc'
SMOKE = True
FULL_DATA = False
RESUME = False
GPUS = 'auto'  # '2' requires both T4s; '1' provides a single-GPU comparison.
PRECISION = 'fp16'  # Use 'fp32' in a new run for an accuracy/performance comparison.
MICROBATCH = 1  # Try 2 in a fresh smoke run if peak VRAM permits.
WORKERS = 2  # Per GPU; use 0 to diagnose DataLoader issues.
MODE = 'smoke' if SMOKE else 'full'
RUN_NAME = f'kaggle_{BACKEND}_{MODE}_{GPUS}gpu_{PRECISION}_mb{MICROBATCH}'
print(BACKEND, MODE, RUN_NAME, 'all data:', FULL_DATA)
""",
    ),
    cell(
        "code",
        """training = [TRAIN_PY, 'scripts/run_kaggle.py', '--backend', BACKEND, '--mode', MODE, '--name', RUN_NAME,
            '--gpus', GPUS, '--precision', PRECISION, '--microbatch', MICROBATCH, '--workers', WORKERS]
if FULL_DATA:
    training += ['--full-data']
if RESUME:
    training += ['--resume']
run(training)
config = json.loads(Path('outputs', RUN_NAME, 'run_config.json').read_text())
print(Path(config['output_dir'], 'report.json').read_text())
""",
    ),
    cell(
        "markdown",
        """## Optional closed-loop simulation

Set RUN_SIMULATION=True to install and run LIBERO after training.
Simulation uses CPU PyTorch and NumPy 1.26 in its own environment (4 GiB install reserve).
The policy stays on GPU in the training environment and serves actions through localhost HTTP.
Simulation subprocesses use MPLBACKEND=Agg without changing the notebook's plotting backend.
Use SKIP_SIM_SETUP=True when the CPU simulator dependencies are already installed.
Optional setup/rollout failures are recorded and allow the result-saving cells to continue.
The small smoke model may have zero success.
Start with 1 initial state/task, then 50 unique states/task for a full evaluation.
""",
    ),
    cell(
        "code",
        """RUN_SIMULATION = False
N_TRIALS = 1  # Set to 50 for a full suite; retain earlier evaluations in a different directory.
SIM_SUITE = 'libero_object'
SKIP_SIM_SETUP = False  # True to reuse .venv-sim-cpu, including after an inline-backend import failure.
RECORD_VIDEO = False  # True saves side-by-side camera MP4s for each trial.
""",
    ),
    cell(
        "code",
        """if RUN_SIMULATION:
    simulation = [TRAIN_PY, 'scripts/run_kaggle_sim.py', '--run', RUN_NAME,
                  '--trials', str(N_TRIALS), '--suite', SIM_SUITE]
    if SKIP_SIM_SETUP:
        simulation += ['--skip-setup']
    if RECORD_VIDEO:
        simulation += ['--record-video']
    result = subprocess.run([str(x) for x in simulation], check=False)
    status_file = Path(config['output_dir'], 'simulation_status.json')
    if status_file.exists():
        print(status_file.read_text())
    if result.returncode:
        print('Simulation failed; inspect the log/status above. Continue saving the training artifacts.')
else:
    print('Simulation skipped. Saved training artifacts:', config['output_dir'])
""",
    ),
    cell(
        "markdown",
        """## Save results

Save the run directory in Kaggle Output before ending the session. ZIP creation is optional
because it duplicates checkpoint/data storage. Each archive contains only the selected run.
For resume in a new session, retain both outputs/RUN_NAME and its data/prepared directory.
Set RESUME=True with exactly the original experiment options. Archives check free disk first.
""",
    ),
    cell(
        "code",
        """SAVE_RESULTS_ZIP = False
SAVE_RESUME_DATA = False
print('Free disk GiB:', shutil.disk_usage('/kaggle/working').free / 2**30)
print('Keep:', PROJECT / config['output_dir'], PROJECT / config['prepared_dir'])
if SAVE_RESULTS_ZIP or SAVE_RESUME_DATA:
    archive = [TRAIN_PY, 'scripts/archive_kaggle.py', '--run', RUN_NAME,
               '--destination', '/kaggle/working']
    if SAVE_RESUME_DATA:
        archive += ['--include-prepared']
    run(archive)
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
    c["id"] = f"robot-foundry-{i:02d}"
    if c["cell_type"] == "code":
        compile("".join(c["source"]), f"cell_{i}", "exec")
Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/kaggle_train.ipynb").write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print("Wrote and syntax-checked Git-clone notebook: notebooks/kaggle_train.ipynb")
