# Tải checkpoint Kaggle và chạy simulation local

Không cần train lại để đánh giá model đã lưu. Pipeline tải policy và metadata của một run, rồi chạy SmolVLA trong môi trường VLA; LIBERO chạy trong môi trường CPU/NumPy 1.26 riêng và gọi policy qua localhost.

## 1. Môi trường VLA và quyền truy cập Kaggle

Chạy từ thư mục gốc repo. Nếu đã có môi trường VLA hoạt động, dùng Python của môi trường đó và bỏ qua bước setup. Ví dụ Windows:

```powershell
python -m pip install uv==0.10.11
uv venv --python 3.12 .venv-vla
uv pip install --python .venv-vla/Scripts/python.exe --no-cache torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv-vla/Scripts/python.exe --no-cache -e . --constraint requirements/train_constraints.txt --no-sources
.venv-vla/Scripts/python.exe -m robot_foundry.cli bootstrap --repos lerobot
uv pip install --python .venv-vla/Scripts/python.exe --no-cache -e "vendor/lerobot[smolvla]" --constraint requirements/train_constraints.txt --no-sources
uv pip install --python .venv-vla/Scripts/python.exe kaggle==2.2.4
```

Linux có thể dùng `python scripts/setup_kaggle.py --env .venv-vla --torch-index cu118` để cài cùng profile rồi cài thêm Kaggle CLI. Installer này dành cho Linux; không dùng nó trên Windows.

CLI Kaggle phải được đăng nhập vào tài khoản có quyền đọc output notebook. Dùng cơ chế đăng nhập/token của Kaggle trên máy; không đặt token trong code, notebook hoặc Git. Repo GitHub public không làm output của notebook private trở thành public.

## 2. Tải đúng run

Thay `OWNER/NOTEBOOK_SLUG` và `RUN_NAME` bằng giá trị của bạn; `RUN_NAME` là tên thư mục trong `robot-vla-foundry/outputs/` của notebook.

```powershell
.venv-vla/Scripts/python.exe scripts/download_kaggle_run.py OWNER/NOTEBOOK_SLUG --run RUN_NAME --destination outputs/RUN_NAME
```

Script cố định version Kaggle ngay khi bắt đầu và tải từng file cần thiết qua API chính thức. Nó không duyệt/tải lại các Python environment được lưu kèm notebook. File được stream vào `.part`, kiểm tra độ dài, rồi thay thế nguyên tử; `download_manifest.json` ghi version, dung lượng và SHA-256. Chạy lại kiểm tra hash trước khi tái dùng file. Khi notebook đã có version mới, dùng destination mới để giữ bản cũ.

Output gồm model đã train, config, train-only stats, data manifest, processor buffers và báo cáo của run. Tokenizer/config VLM được khôi phục từ revision public đã khóa trong registry; trọng số policy vẫn là trọng số tải từ Kaggle. Optimizer và prepared data không nằm trong download này; đây là bộ artifact phục vụ inference, chưa đủ để resume training. Giữ ít nhất 4 GiB disk trống trước download.

## 3. Chạy LIBERO và ghi video

```powershell
.venv-vla/Scripts/python.exe scripts/run_kaggle_sim.py --run RUN_NAME --trials 1 --suite libero_object --record-video
```

Lần đầu script cài `.venv-sim-cpu`; installer cần dự phòng 4 GiB disk. Nếu môi trường đã cài xong, thêm `--skip-setup`; có thể chọn venv khác nằm trong project bằng `--sim-env TEN_VENV`. Linux dùng đường dẫn `.venv-vla/bin/python` và cần thư viện render EGL/OpenGL theo [`KAGGLE.md`](KAGGLE.md).

SmolVLA inference đóng băng toàn bộ tham số và dùng FP16 trên CUDA. Chế độ này đã chạy được với checkpoint H50/256 px trên RTX A1000 Laptop 4 GB; dung lượng thực tế còn phụ thuộc model, input và các process đang dùng GPU. Điều kiện tối thiểu 14 GiB của profile **training** vẫn giữ nguyên. Inference không tạo optimizer hoặc gradient. Khi gọi `scripts/policy_server.py` trực tiếp, `--device cpu --precision fp32` cho phép inference CPU; runner tự động ở trên ưu tiên GPU và precision đã lưu trong run config.

Kết quả nằm trong:

```text
outputs/RUN_NAME/simulation_status.json
outputs/RUN_NAME/policy_server.log
outputs/libero_RUN_NAME_libero_object_1/report.json
outputs/libero_RUN_NAME_libero_object_1/trials.jsonl
outputs/libero_RUN_NAME_libero_object_1/videos/task_00_state_00.mp4
```

Video có camera ngoài ở trái, camera cổ tay ở phải, xoay giống input policy; 20 FPS theo thời gian mô phỏng. Đường dẫn video được ghi trong report và mỗi trial. Runner tắt policy server do nó tạo sau evaluation và giữ checkpoint khi simulation lỗi.

`--trials 1` chạy 10 tác vụ Object, mỗi tác vụ một initial state chính thức, tối đa 300 bước và thực thi 4 action trước khi dự đoán chunk mới. Đây là kiểm tra nhỏ; full suite cần `--trials 50` và 500 lượt. Dùng thư mục evaluation mới hoặc giữ/chuyển kết quả cũ trước khi chạy lại cùng suite/số trials. `status=complete` nghĩa simulation đã chạy xong; tỷ lệ thành công phải đọc từ `successes`, `trials` và `macro_success_rate`. Xem [`BENCHMARK.md`](BENCHMARK.md) để hiểu giới hạn đánh giá.
