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
The default is pretrained SmolVLA on one GPU with >=14 GiB VRAM. GR00T has a separate >=40 GB Linux recipe.
Run the 20-update smoke experiment first; then use a fresh name for full training.
This notebook is syntax-checked and its policy/data components are tested locally;
full Kaggle training depends on the allocated hardware and network.
""",
    ),
    cell(
        "code",
        """from pathlib import Path
import os, sys, subprocess, shutil, json

REPO_URL = 'https://github.com/TrgPhan/robot-vla-foundry.git'
GIT_REF = 'kaggle-v1.1'  # Pinned release. Choose 'main' in a fresh directory for later changes.
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

SMOKE=True trains 20 optimizer updates. SMOKE=False uses the full 5,000-update recipe.
FULL_DATA=True selects all public Object/Spatial episodes; the default caps at 10 episodes/task.
Keep mode/backend/name/full-data options unchanged when resuming an interrupted run.
T4/P100 use FP32; CUDA 11.8 wheels are pinned to retain older GPU support.
""",
    ),
    cell(
        "code",
        """BACKEND = 'smolvla'  # or 'tiny_bc'
SMOKE = True
FULL_DATA = False
RESUME = False
MODE = 'smoke' if SMOKE else 'full'
RUN_NAME = f'kaggle_{BACKEND}_{MODE}'
print(BACKEND, MODE, RUN_NAME, 'all data:', FULL_DATA)
""",
    ),
    cell(
        "code",
        """training = [TRAIN_PY, 'scripts/run_kaggle.py', '--backend', BACKEND, '--mode', MODE, '--name', RUN_NAME]
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
Simulation uses NumPy 1.26 in its own environment. SmolVLA stays in the training environment
and serves actions through a loopback HTTP endpoint. The small smoke model may have zero success.
Start with 1 initial state/task, then 50 unique states/task for a full evaluation.
""",
    ),
    cell(
        "code",
        """RUN_SIMULATION = False
SIM_PY = PROJECT / '.venv-sim/bin/python'
checkpoint = (PROJECT / config['output_dir'] / 'best').resolve()
server = None
server_log = None
if RUN_SIMULATION:
    run([TRAIN_PY, 'scripts/setup_sim.py', '--env', PROJECT / '.venv-sim', '--python', '3.12'])
if RUN_SIMULATION and BACKEND == 'smolvla':
    server_log = open('outputs/policy_server.log', 'w')
    server = subprocess.Popen([str(TRAIN_PY), 'scripts/policy_server.py', '--checkpoint', str(checkpoint),
                               '--backend', 'smolvla', '--port', '8765'], cwd=PROJECT,
                              stdout=server_log, stderr=subprocess.STDOUT)
    try:
        run([TRAIN_PY, 'scripts/wait_policy_server.py', '--port', '8765'])
    except Exception:
        server.terminate()
        server.wait(timeout=30)
        server_log.close()
        raise
""",
    ),
    cell(
        "code",
        """N_TRIALS = 1  # Set to 50 for a full suite; use a fresh output directory each time.
evaluation_output = f'outputs/libero_{RUN_NAME}_{N_TRIALS}'
evaluation = [SIM_PY, '-m', 'robot_foundry.cli', 'eval-libero', '--checkpoint', checkpoint,
              '--backend', BACKEND, '--suite', 'libero_object', '--episodes-per-task', N_TRIALS,
              '--max-steps', '300', '--execution-horizon', '4', '--output', evaluation_output]
if BACKEND == 'smolvla':
    evaluation += ['--policy-url', 'http://127.0.0.1:8765']
try:
    if RUN_SIMULATION:
        run(evaluation)
finally:
    if server:
        server.terminate()
        server.wait(timeout=30)
    if server_log:
        server_log.close()
if RUN_SIMULATION:
    print(Path(evaluation_output, 'report.json').read_text())
""",
    ),
    cell(
        "markdown",
        """## Save results

Download robot_foundry_results.zip from Kaggle Output before ending the session.
For resume in a new session, also save the prepared data archive and restore both
archives into PROJECT. Set RESUME=True with exactly the original experiment options.
""",
    ),
    cell(
        "code",
        """shutil.make_archive('/kaggle/working/robot_foundry_results', 'zip', root_dir=PROJECT, base_dir='outputs')
print('/kaggle/working/robot_foundry_results.zip')
SAVE_RESUME_DATA = False
if SAVE_RESUME_DATA:
    shutil.make_archive('/kaggle/working/robot_foundry_prepared', 'zip', root_dir=PROJECT, base_dir='data/prepared')
    print('/kaggle/working/robot_foundry_prepared.zip')
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
