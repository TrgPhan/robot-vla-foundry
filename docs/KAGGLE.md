# Kaggle T4 ×2: train và infer VLA toàn thân G1

Import [`kaggle_g1_train.ipynb`](../notebooks/kaggle_g1_train.ipynb) vào Kaggle; bật **Internet** và **GPU T4 ×2**. [`kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb) là bản giống hệt. Notebook clone repo public ở tag `g1-vla-v1`, không cần GitHub token.

Notebook Panda/LIBERO cũ nằm riêng ở [`kaggle_panda_train.ipynb`](../notebooks/kaggle_panda_train.ipynb); hướng dẫn cũ được giữ tại [`KAGGLE_PANDA.md`](KAGGLE_PANDA.md).

## Các bước và dấu hiệu nhận biết

| Bước | Việc thực hiện | Kết quả |
|---|---|---|
| Setup | Cài môi trường train PyTorch/LeRobot và môi trường SONIC/MuJoCo riêng | Doctor báo GPU, simulator import thành công |
| `COLLECT` | SONIC teacher chạy physics; ghi RGB/state70/token64 đồng bộ | Manifest, stats train-only, paired episodes |
| `TRAIN` | SmolVLA forward/loss/backward/optimizer, validation và test | `optimizer_updates > 0`, checkpoint G1 64D |
| Trained simulation | Nạp checkpoint vừa train; VLA dự đoán token cho SONIC decoder | MP4 toàn thân, trials, token traces |
| Baseline | Chạy cùng test cases bằng token trung bình của train | Report riêng để đối chiếu lợi ích policy |

`COLLECT` là chuẩn bị dữ liệu, chưa train. Bước train có optimizer cập nhật action expert và ba state/action projections; VLM và SONIC frozen. Sau khởi tạo pose, inference không gọi reference encoder hoặc dùng teacher fallback.

## Cấu hình chạy

Giữ `SMOKE=True`, `MICROBATCH=1`, `WORKERS=2`, `GPUS='auto'` cho lần đầu. Smoke chạy **20 optimizer updates** và kiểm tra export/inference; chưa nhằm học hội tụ. Khi smoke chạy ổn, đổi `SMOKE=False`: tên run chuyển sang `full`, recipe chạy **5.000 bước**. Muốn resume run bị ngắt, dùng lại đúng options/data và `RESUME=True`.

Full training giữ effective batch **16 = 2 GPU × microbatch1 × accumulation8**; một GPU dùng accumulation16. Smoke dùng accumulation2 để kiểm tra nhanh. FP16 autocast/GradScaler, DDP một process/GPU, fused AdamW và cache PNG lossless giảm chi phí. Hai T4 không cộng VRAM. Chỉ tăng microbatch sau một smoke mới đo đủ VRAM; mỗi GPU vẫn phải chứa model/gradient riêng.

Default collection tạo 24 episodes ở 256 px/H20: hai motion primitives, mỗi primitive 8 train/2 val/2 test seeds. Đây là đi bộ/cử động toàn thân với bàn tay mesh cố định; chưa có gắp vật. Val/test đổi scenario nhưng dùng cùng primitive nguồn. Xem cơ chế, nguồn dữ liệu và giới hạn tại [`G1_TRAINING.md`](G1_TRAINING.md).

## Giữ output và xử lý lỗi

Lưu cả `outputs/RUN_NAME/`, `data/prepared/DATASET_NAME/` và `outputs/g1_eval_RUN_NAME/`. Checkpoint `best/` dùng inference; `training_state.pt` chứa optimizer/scheduler/scaler/RNG để resume. Model được stage một lần; cần ít nhất 6 GiB disk dự phòng sau staging. Export được ghi vào thư mục tạm rồi thay thế để giữ checkpoint cũ nếu ghi thất bại. Notebook không tạo ZIP trùng model/data mặc định.

Inference lỗi được ghi ở `outputs/RUN_NAME/g1_simulation_status.json`; baseline ghi `g1_mean_baseline_status.json`. Notebook giữ artifacts train và tiếp tục tới cell hiển thị/lưu output. Không diễn giải video tồn tại hoặc process hoàn tất thành task success; đọc report của đúng run.

Nếu hết disk, kiểm tra `df -h /kaggle/working` và `du -sh data/* outputs/* /kaggle/working/*`. Giữ checkpoint đang dùng và prepared data đúng fingerprint; xem [`KAGGLE_PANDA.md`](KAGGLE_PANDA.md) cho quy trình recovery chung. Không dùng launcher recovery Panda để tạo recipe G1 khác. Với G1, sau khi đủ disk hãy chạy lại `run_g1_train.py` với cùng options và `--resume`.

Full T4 ×2 throughput/VRAM và chất lượng sau 5.000 bước chưa được đo trên Kaggle. Local đã kiểm chứng optimizer SmolVLA G1, save/reload và inference toàn thân; báo cáo ngắn đó là kiểm chứng tích hợp, chưa chứng minh model học tốt hơn baseline.
