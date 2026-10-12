# Clone GitHub và chạy train trên Kaggle

Repo public: **https://github.com/TrgPhan/robot-vla-foundry**.

Bạn không cần GitHub token. Tạo Kaggle Notebook, bật **Internet** và chọn **GPU T4 ×2** trong Settings. Notebook hoàn chỉnh nằm ở [`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb): tải file từ GitHub rồi import vào Kaggle, hoặc dùng các cell dưới đây.

## Cell 1 — clone release

```python
import os
import subprocess
from pathlib import Path

PROJECT = Path('/kaggle/working/robot-vla-foundry')
subprocess.run([
    'git', 'clone', '--depth', '1', '--branch', 'kaggle-v2',
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
    '--backend', 'smolvla', '--mode', 'smoke', '--name', 'smolvla_smoke',
    '--gpus', '2', '--precision', 'fp16'
], check=True)
```

Luồng chạy: kiểm tra GPU → tải/chuẩn bị dữ liệu LIBERO → tạo cache PNG lossless → stage model/assets một lần → `torchrun` hai process → train 20 bước → validation → lưu checkpoint → test open-loop. Số optimizer updates thực tế và bước bỏ qua do AMP overflow được ghi riêng. Data cap là 10 episode/task. Output:

```text
outputs/smolvla_smoke/
  run_config.json       chính xác recipe đã dùng
  git_provenance.json   commit và phần cứng
  training_state.pt     optimizer/scheduler/GradScaler/RNG từng rank để resume
  metrics.jsonl
  report.json
  best/                 model + stats + tokenizer/assets
data/prepared/smolvla_smoke/
  manifest.json
  stats.json
  episodes/
  packed/               PNG blob + mmap metadata; pixel không đổi
```

Nếu không có GPU phù hợp, đổi `--backend tiny_bc`; backend đó là baseline từ đầu, không phải foundation VLA.

## Cell 4 — full training khi smoke pass

```python
subprocess.run([
    TRAIN_PY, 'scripts/run_kaggle.py',
    '--backend', 'smolvla', '--mode', 'full', '--full-data', '--name', 'smolvla_full',
    '--gpus', '2', '--precision', 'fp16'
], check=True)
```

Recipe full dùng 5.000 bước, microbatch 1/GPU, accumulation 8 trên hai GPU: effective batch 16. `--gpus 1` dùng accumulation 16; `--gpus auto` chọn tối đa hai GPU. FP16 có GradScaler, loss/reduction giữ FP32, gradient clip một lần sau accumulation, DDP chỉ đồng bộ ở microstep cuối. Hai worker prefetch/GPU, pinned memory và fused AdamW giảm overhead. `--full-data` lấy toàn bộ Object + Spatial; full-data và packed cache cần thêm disk/thời gian chuẩn bị. Mỗi run có split/stats manifest riêng.

Validation chọn checkpoint trên tối đa 64 sample cố định, chia đều không đếm trùng; test cuối dùng toàn bộ test split. Metric ghi rõ sample cap. `metrics.jsonl` có `train_samples_per_second`, `train_seconds_per_step`, `peak_cuda_memory_mib_by_rank`, `grad_scale`, `amp_skipped_updates`. Throughput loại thời gian validation/export, nhưng cửa sổ đầu có startup nên dùng cửa sổ sau để so sánh. Chưa có số đo tốc độ hoặc success rate của full run trên Kaggle.

Đo một GPU so với hai GPU với cùng batch 16:

```python
for n in ('1', '2'):
    subprocess.run([
        TRAIN_PY, 'scripts/run_kaggle.py', '--backend', 'smolvla',
        '--mode', 'benchmark', '--gpus', n, '--precision', 'fp16',
        '--name', f'benchmark_{n}gpu'
    ], check=True)
```

Benchmark 100 bước giữ cùng recipe dữ liệu và effective batch; dùng hàng `step=100` để so sánh throughput, VRAM và AMP skips. Không giả định tốc độ tăng 2×. Có thể thử `--microbatch 2` trong smoke mới rồi benchmark mới; full/benchmark tự giảm accumulation tương ứng.

## Tiếp tục sau khi bị ngắt

Trong cùng phiên, chạy lại đúng flags kèm `--resume`:

```python
subprocess.run([
    TRAIN_PY, 'scripts/run_kaggle.py',
    '--backend', 'smolvla', '--mode', 'full', '--full-data',
    '--name', 'smolvla_full', '--gpus', '2', '--precision', 'fp16', '--resume'
], check=True)
```

Ở phiên mới, restore **cả output và prepared data** của run trước vào project. Giữ nguyên số GPU, precision, workers, microbatch và các flags cũ khi resume. Checkpoint v1 FP32 không resume sang engine v2; tạo run mới. Code kiểm tra config/world-size/fingerprint. Packed cache thiếu có thể xây lại lossless từ các episode NPZ; không thay đổi dữ liệu mẫu.

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
