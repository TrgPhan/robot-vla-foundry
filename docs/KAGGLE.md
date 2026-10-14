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
    'git', 'clone', '--depth', '1', '--branch', 'kaggle-v2.5',
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

Luồng chạy: kiểm tra GPU → tải/chuẩn bị LIBERO, cache PNG lossless theo từng episode và bỏ NPZ sau khi so sánh pixel/nhãn → stage model/assets một lần → kiểm tra disk dự phòng → `torchrun` hai process → train 20 bước → validation → lưu checkpoint → test open-loop. Số optimizer updates thực tế và bước bỏ qua do AMP overflow được ghi riêng. Data cap là 10 episode/task. Output:

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
  episodes/             provenance JSON; NPZ có thể đã compact
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

Ở phiên mới, restore **cả output và prepared data** của run trước vào project. Giữ nguyên số GPU, precision, workers, microbatch và các flags cũ khi resume. Checkpoint v1 FP32 không resume sang engine v2; tạo run mới. Code kiểm tra config/world-size/fingerprint. Sau compact, phải giữ `packed/`, manifest/stats và provenance JSON; NPZ đã bỏ thì không thể dùng riêng manifest để train. Nếu còn NPZ, cache mất có thể xây lại lossless.

Ưu tiên giữ thư mục output và prepared data trong Kaggle Output. ZIP tạo thêm một bản dữ liệu nên mặc định tắt. Khi đủ disk, archive riêng một run:

```python
subprocess.run([
    TRAIN_PY, 'scripts/archive_kaggle.py', '--run', 'smolvla_full',
    '--destination', '/kaggle/working', '--include-prepared'
], check=True)
```

Archive kiểm tra free disk và ghi qua file tạm; archive cũ còn nguyên nếu ghi thất bại. `training_state.pt` lớn vì chứa model và optimizer; source Git không lưu dữ liệu/checkpoints. Launcher giữ ít nhất 6 GiB trống sau khi stage SmolVLA; trước từng lần lưu, engine tính thêm dung lượng tensor/optimizer và phần ghi tạm. Không suy ra disk budget từ VRAM.

## Khôi phục `No space left on device` từ v2

Lỗi Safetensors ở bước export là thiếu disk. Giữ nguyên `MICROBATCH`, workers, dataset cap và seed khi tiếp tục run cũ. Dừng process train trước khi chạy cleanup; lệnh dưới đây chỉ xử lý run được chọn, không bỏ checkpoint hợp lệ hay thư mục của các run khác.

Nếu disk đã đầy tới mức `git fetch` cũng không chạy được, giải phóng **chỉ file export đã được chứng minh lỗi** bằng môi trường train đã có:

```python
import os, subprocess
os.chdir('/kaggle/working/robot-vla-foundry')
TRAIN_PY = '/kaggle/working/robot-vla-env/bin/python'
RUN = 'kaggle_smolvla_full_2gpu_fp16'  # Tên run cũ, không thêm hậu tố mb của notebook mới.
subprocess.run([TRAIN_PY, '-c', '''
from pathlib import Path
from safetensors import SafetensorError, safe_open
import sys
p = Path('outputs') / sys.argv[1] / 'best/model.safetensors'
if p.exists():
    try:
        with safe_open(p, framework='pt', device='cpu') as f:
            list(f.keys())
        print('Readable weights retained:', p)
    except SafetensorError as error:
        if 'I/O error' in str(error):
            raise
        p.unlink()
        print('Removed truncated export:', p)
''', RUN], check=True)
```

Cập nhật source tại chỗ, không clone thêm môi trường/data và không chạy lại setup:

```python
subprocess.run(['git', 'fetch', '--depth', '1', 'origin', 'tag', 'kaggle-v2.3'], check=True)
subprocess.run(['git', 'checkout', 'kaggle-v2.3'], check=True)
subprocess.run([
    TRAIN_PY, 'scripts/recover_kaggle.py', '--run', RUN,
    '--apply', '--prune-raw', '--train'
], check=True)
```

Recovery đọc nguyên `outputs/RUN/run_config.json`, so sánh cache với NPZ rồi bỏ bản trùng; `--prune-raw` bỏ chỉ raw parquet/video có hash khớp provenance, có thể tải lại từ revision đã khóa. Model/optimizer checkpoint hợp lệ được giữ. File tạm optimizer đọc hoàn chỉnh và đúng provenance được cứu nếu mới hơn checkpoint hiện tại; file ghi dở được bỏ. Có checkpoint thì resume; không có checkpoint thì restart optimizer với prepared data, không decode/tải dữ liệu train lại. Model nền/tokenizer có thể tải lại nếu Hub cache không còn.

V2 đã ghi đè trực tiếp `best`, nên nếu export cũ bị hỏng thì không thể khôi phục chính xác model best trước đó chỉ từ metric. Engine sẽ re-score model trong optimizer checkpoint và dùng làm best khởi đầu, ghi `best_reconstructed_from_resume=true` trong report. Nếu chưa có optimizer checkpoint, phần train chưa lưu phải chạy lại. V2.1 export vào thư mục tạm và giữ best cũ khi lưu thất bại.

Nếu sau compact vẫn chưa đủ 6 GiB, recovery dừng trước train. Xem `df -h` và `du -sh` để kiểm tra môi trường, archive và các run khác; công cụ không tự xóa các run khác. Run mới có thể bắt đầu với `FULL_DATA=False`; đổi dataset cap của run cũ sẽ làm mất khả năng resume đúng recipe.

## Simulation sau train

Notebook có các cell riêng: đổi `RUN_SIMULATION=True` để chạy LIBERO trong `.venv-sim-cpu`, dùng PyTorch CPU và NumPy 1.26. Policy vẫn chạy GPU trong môi trường train và phục vụ qua localhost HTTP. Installer giữ dự phòng 4 GiB cho CPU, 8 GiB nếu chủ động chọn CUDA; đây là budget bảo thủ, không phải số đo trên Kaggle. `uv --no-cache` tránh giữ thêm bản unpacked wheel sau cài đặt. Simulation lỗi được ghi vào `outputs/RUN_NAME/simulation_status.json` và notebook tiếp tục tới cell lưu kết quả. Mặc định simulation tắt để `Run All` ưu tiên train. Metric open-loop không phải tỷ lệ thành công; protocol benchmark ở [`BENCHMARK.md`](BENCHMARK.md).

V2.4 có `RECORD_VIDEO=True` để lưu MP4 hai camera của từng lượt; CLI tương ứng là `--record-video`. Hướng dẫn tải policy từ output Kaggle và chạy trên Windows/Linux local ở [`LOCAL_SIM.md`](LOCAL_SIM.md).

GR00T N1.7 + SONIC vẫn là nhánh riêng, cần GPU lớn và paired G1 data; notebook Kaggle này không giả lập một model GR00T khi đang dùng SmolVLA.

## Hết disk khi cài `.venv-sim` sau train

Lỗi copy `libtorch_cuda.so` trong `scripts/setup_sim.py` thuộc bước cài simulation. Không đổi microbatch và không chạy lại train chỉ vì lỗi này. Nếu `report.json` có số bước bằng recipe và `best/model.safetensors` đọc được, lượt train đã có output cuối; lỗi simulation không phải lỗi optimizer. Nếu phiên đã reset và chưa lưu output, cần bản sao checkpoint để tiếp tục.

Trước hết kiểm tra các report còn trong phiên:

```python
from pathlib import Path
import json, os, subprocess, shutil

PROJECT = Path('/kaggle/working/robot-vla-foundry').resolve()
TRAIN_PY = '/kaggle/working/robot-vla-env/bin/python'
os.chdir(PROJECT)
for path in sorted((PROJECT / 'outputs').glob('*/report.json')):
    report = json.loads(path.read_text())
    print(path.parent.name, 'steps:', report.get('steps'), 'test:', report.get('test'))
```

Giữ tên run đã in ở trên. Cell sau chỉ tái tạo môi trường simulation và bỏ cache wheel có thể tải lại; checkpoint, output và prepared data được giữ. Không chạy cell khi process simulation đang dùng `.venv-sim`.

```python
RUN = RUN_NAME  # Hoặc gán chính xác tên thư mục outputs của lượt train cũ.
old_sim = PROJECT / '.venv-sim'
if old_sim.exists():
    if old_sim.is_symlink() or old_sim.resolve().parent != PROJECT:
        raise ValueError('Simulation environment must stay inside PROJECT.')
    if not (old_sim / 'pyvenv.cfg').is_file():
        raise ValueError('Refusing to remove a directory that is not a virtual environment.')
    shutil.rmtree(old_sim)
subprocess.run(['uv', 'cache', 'clean', 'torch', 'torchvision'], check=True)
subprocess.run(['git', 'fetch', '--depth', '1', 'origin', 'tag', 'kaggle-v2.3'], check=True)
subprocess.run(['git', 'checkout', 'kaggle-v2.3'], check=True)
result = subprocess.run([
    TRAIN_PY, 'scripts/run_kaggle_sim.py', '--run', RUN,
    '--trials', '1', '--suite', 'libero_object'
], check=False)
status = PROJECT / 'outputs' / RUN / 'simulation_status.json'
if status.exists():
    print(status.read_text())
print('Simulation return code:', result.returncode)
```

`run_kaggle_sim.py` đọc config/checkpoint đã lưu và chỉ chạy setup/evaluation. Nếu vẫn thiếu 4 GiB, compact và prune raw theo hash của đúng run bằng `scripts/kaggle_storage.py --run RUN --apply --prune-raw`, rồi chỉ chạy lại simulation. Khi `.venv-sim-cpu` đã cài xong, thêm `--skip-setup` để tái dùng môi trường. Một evaluation hoàn tất với 0% success vẫn là kết quả hợp lệ; trạng thái `complete` của simulation không có nghĩa model đã đạt success cao.

Linux headless dùng `MUJOCO_GL=egl` và `PYOPENGL_PLATFORM=egl` trước khi import MuJoCo/Robosuite. Trên máy Linux mới thiếu thư viện OpenGL/EGL, cài `libegl1` và `libgl1` bằng package manager hệ thống; wheel PyTorch CPU không cung cấp các thư viện render này. CI kiểm tra cài mới trên Ubuntu với hai thư viện đó. Môi trường GPU Kaggle vẫn cần driver/render backend phù hợp để chạy rollout.

## Lỗi Matplotlib inline backend khi simulation đã cài xong

Kaggle kernel có thể export `MPLBACKEND=module://matplotlib_inline.backend_inline`. Process Python của simulator kế thừa biến này, nhưng môi trường NumPy 1.26 không cài backend notebook đó. V2.3 dùng `Agg` riêng cho process setup, policy server và rollout; backend hiển thị của notebook được giữ nguyên. CI tái hiện môi trường có biến inline trước khi kiểm tra cài mới.

Nếu log đã cài `robot-foundry==0.2.2` rồi chỉ lỗi tại `import matplotlib`, có thể tái dùng môi trường đã cài mà không cần tải lại package hoặc train. Cell này cũng dùng được với code v2.2; nó truyền `Agg` vào toàn bộ process con:

```python
import os, subprocess
from pathlib import Path

PROJECT = Path('/kaggle/working/robot-vla-foundry')
TRAIN_PY = '/kaggle/working/robot-vla-env/bin/python'
SIM_PY = PROJECT / '.venv-sim-cpu/bin/python'
RUN = 'kaggle_smolvla_full_autogpu_fp16_mb1'  # Tên chính xác trong log; đổi nếu run của bạn khác.
sim_env = dict(os.environ, MPLBACKEND='Agg')
subprocess.run([
    str(SIM_PY), '-c', 'import matplotlib; print(matplotlib.get_backend())'
], env=sim_env, check=True)
subprocess.run([
    TRAIN_PY, 'scripts/run_kaggle_sim.py', '--run', RUN,
    '--trials', '1', '--suite', 'libero_object', '--skip-setup'
], cwd=PROJECT, env=sim_env, check=True)
```

Bắt đầu với 1 initial state/task. Khi lượt đó hoàn tất, chọn 50 cho evaluation đầy đủ nếu thời gian phiên còn đủ. Các output có số trials khác nhau nằm ở các thư mục khác nhau; giữ kết quả cũ khi chạy lại cùng số trials. Notebook v2.3 có `SKIP_SIM_SETUP=True` cho trường hợp tái dùng môi trường này. Với khoảng 1.28 GiB free disk như log đã gửi, dùng `--skip-setup` để tránh cài lại và giữ ZIP tắt; code đánh giá này không chạy train và không xóa checkpoint/data.
