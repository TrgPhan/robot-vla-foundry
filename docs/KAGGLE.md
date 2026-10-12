# Clone GitHub và chạy train trên Kaggle

Repo public: **https://github.com/TrgPhan/robot-vla-foundry**.

Bạn không cần GitHub token. Tạo Kaggle Notebook, bật **Internet** và chọn **GPU** trong Settings. Notebook hoàn chỉnh nằm ở [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb): tải file từ GitHub rồi import vào Kaggle, hoặc dùng các cell dưới đây.

## Cell 1 — clone release

```python
import os
import subprocess
from pathlib import Path

PROJECT = Path('/kaggle/working/robot-vla-foundry')
subprocess.run([
    'git', 'clone', '--depth', '1', '--branch', 'kaggle-v1',
    'https://github.com/TrgPhan/robot-vla-foundry.git', str(PROJECT)
], check=True)
os.chdir(PROJECT)
```

Cell ngắn này dành cho phiên mới; notebook đầy đủ kiểm tra và tái sử dụng checkout nếu cell đã chạy. Muốn lấy code mới hơn, dùng `main` trong một thư mục mới. Release tag giúp lưu đúng source của experiment.

## Cell 2 — cài môi trường train

```python
import sys
subprocess.run([sys.executable, 'scripts/setup_kaggle.py'], check=True)
TRAIN_PY = '/kaggle/working/robot-vla-env/bin/python'
subprocess.run([TRAIN_PY, '-m', 'robot_foundry.cli', 'doctor'], check=True)
```

Script tạo Python 3.12 riêng, cài PyTorch 2.7.1/cu118 và LeRobot đúng commit. CUDA wheel được khóa cả sau khi cài LeRobot. Dataset/weights tải từ public Hugging Face repositories. SmolVLA profile cần khoảng 14 GiB VRAM trở lên và phải qua smoke để đo bộ nhớ thực tế; hai T4 không được cộng VRAM.

## Cell 3 — smoke train thật

```python
subprocess.run([
    TRAIN_PY, 'scripts/run_kaggle.py',
    '--backend', 'smolvla', '--mode', 'smoke', '--name', 'smolvla_smoke'
], check=True)
```

Luồng chạy: kiểm tra GPU → tải/chuẩn bị dữ liệu LIBERO → train 20 optimizer updates → validation → lưu checkpoint → test open-loop. Data cap là 10 episode/task. Output:

```text
outputs/smolvla_smoke/
  run_config.json       chính xác recipe đã dùng
  git_provenance.json   commit và phần cứng
  training_state.pt     optimizer/scheduler/RNG để resume
  metrics.jsonl
  report.json
  best/                 model + stats + tokenizer/assets
data/prepared/smolvla_smoke/
  manifest.json
  stats.json
  episodes/
```

Nếu không có GPU phù hợp, đổi `--backend tiny_bc`; backend đó là baseline từ đầu, không phải foundation VLA.

## Cell 4 — full training khi smoke pass

```python
subprocess.run([
    TRAIN_PY, 'scripts/run_kaggle.py',
    '--backend', 'smolvla', '--mode', 'full', '--full-data', '--name', 'smolvla_full'
], check=True)
```

Recipe full dùng 5.000 optimizer updates, microbatch 1, accumulation 16; `--full-data` lấy toàn bộ Object + Spatial. Mỗi run có split/stats manifest riêng, tránh ghi đè manifest của experiment khác. Full-data tốn thêm thời gian tải, decode và disk. Không có cam kết thời gian chạy hoặc success rate trên GPU Kaggle chưa được thực thi.

## Tiếp tục sau khi bị ngắt

Trong cùng phiên, chạy lại đúng flags kèm `--resume`:

```python
subprocess.run([
    TRAIN_PY, 'scripts/run_kaggle.py',
    '--backend', 'smolvla', '--mode', 'full', '--full-data',
    '--name', 'smolvla_full', '--resume'
], check=True)
```

Ở phiên mới, restore **cả output và prepared data** của run trước vào project. Chỉ checkpoint model không đủ để resume chính xác nếu dữ liệu/config thay đổi. Code kiểm tra fingerprint và không tự bỏ qua mismatch.

Để tải kết quả về:

```python
import shutil
shutil.make_archive('/kaggle/working/robot_foundry_results', 'zip', root_dir=PROJECT, base_dir='outputs')
shutil.make_archive('/kaggle/working/robot_foundry_prepared', 'zip', root_dir=PROJECT, base_dir='data/prepared')
```

Lưu hai file trong Kaggle Output trước khi kết thúc phiên. `training_state.pt` có thể lớn vì chứa optimizer; source Git không lưu dữ liệu hoặc checkpoints.

## Simulation sau train

Notebook có các cell riêng: đổi `RUN_SIMULATION=True` để chạy LIBERO với môi trường NumPy 1.26 tách biệt. SmolVLA được phục vụ qua localhost HTTP từ môi trường train. Mặc định simulation tắt để `Run All` ưu tiên train. Metric open-loop không phải tỷ lệ thành công; protocol benchmark ở [`BENCHMARK.md`](BENCHMARK.md).

GR00T N1.7 + SONIC vẫn là nhánh riêng, cần GPU lớn và paired G1 data; notebook Kaggle này không giả lập một model GR00T khi đang dùng SmolVLA.

