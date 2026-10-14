# G1/SONIC: full-body simulation và checkpoint Panda

Checkpoint notebook `kaggle_train.ipynb` là SmolVLA cho **Panda single-arm manipulation**. LIBERO dùng state 8D, hai camera và OSC action 7D; không có chân, floating base hoặc SONIC. Video trước đúng với Panda, nhưng nhánh đó chưa đáp ứng mục tiêu humanoid. Report LIBERO hiện ghi `robot=franka_panda`, `whole_body_control=false`, `sonic_used=false`.

Nhánh mới chạy **SONIC v1.1 pretrained trên G1 29 DOF** trong MuJoCo bằng ONNX Runtime CPU. Camera ngoài thấy cả thân và chân. Đây là inference controller và dynamics có tiếp xúc sàn, đầu vào motion tham chiếu NVIDIA; chưa có VLA vision/language hoặc tác vụ gắp vật. Bàn tay của model 29 DOF là mesh cố định; Dex3 finger actuation và object manipulation cần scene/controller riêng.

| Nhánh | Điều khiển | Kiểm chứng |
|---|---|---|
| Kaggle SmolVLA/LIBERO | Panda OSC 7D | Arm train/inference |
| `run_fullbody_sim.py` | Token 64D → 29 action → PD torque | G1 free-base chạy thật, có MP4/telemetry |
| GR00T G1/SONIC | Token 64D + hai tay 7D mỗi tay | Validator/launcher; cần paired G1 data/checkpoint |

## Chạy Windows

Từ thư mục gốc repo:

```powershell
python -m pip install uv==0.10.11
python scripts/setup_fullbody_sim.py
.venv-fullbody/Scripts/python.exe scripts/run_fullbody_sim.py --output outputs/g1_sonic_fullbody
```

Installer cần Git và ít nhất 2 GiB disk dự phòng, tạo `.venv-fullbody` với `requirements/fullbody_sim.txt`, checkout SONIC đúng commit và kiểm tra import không kéo PyTorch. Không cần Isaac Lab/GPU cho inference ONNX. Profile này dùng `--no-deps` khi cài mã project; không dùng venv này để train VLA.

Linux dùng `.venv-fullbody/bin/python`. Renderer cần EGL/OpenGL; headless có GPU thường dùng `MUJOCO_GL=egl`. CPU rendering dùng `libosmesa6` cùng `MUJOCO_GL=osmesa`, `PYOPENGL_PLATFORM=osmesa`. CI chạy profile CPU/OSMesa với encoder/decoder thật và kiểm tra MP4 rollout ngắn.

Hai ONNX SONIC v1.1 tải từ `nvidia/GEAR-SONIC` đúng revision registry. Meshes và clip `walking_quip_360_R_002__A428` đọc đúng commit upstream; Git LFS files được resolve và kiểm tra SHA-256. Cache ở `data/models/sonic_mujoco/`, không đưa model/dataset vào Git.

```text
outputs/g1_sonic_fullbody/fullbody.mp4
outputs/g1_sonic_fullbody/report.json
outputs/g1_sonic_fullbody/telemetry.jsonl
outputs/g1_sonic_fullbody/simulation_status.json
```

Chọn output mới khi chạy lại. `--seconds 2` cho lượt ngắn; clip mặc định 9,1 giây. Không lặp clip âm thầm để tăng thời gian đánh giá. Video gồm 0,5 giây giữ pose đầu rồi tracking, tổng 9,6 giây ở 50 FPS theo thời gian simulation; đây không phải cam kết realtime cho toàn pipeline.

## Cơ chế đối chiếu với NVIDIA

1. Khởi tạo root và 29 khớp từ frame đầu. Sau đó torque thay đổi trạng thái qua `mj_step`, không chép pose tham chiếu vào robot từng frame, không bật elastic band.
2. Encoder mode G1 là scalar `0` ở slot đầu vector 4D, không phải one-hot. Encoder 1.751D nhận 10 frame khớp/vận tốc cách nhau 5 bước và root orientation theo heading robot; modality không thuộc mode G1 được zero như upstream.
3. Encoder tạo token 64D. Decoder 994D nhận token và history 10 frame từ cũ đến mới: angular velocity, joint positions trừ default pose, joint velocities, action trước, projected gravity.
4. Hai permutation inverse đổi ordering Isaac và MuJoCo/hardware. Action scale/default pose/Kp/Kd theo `policy_parameters.hpp`; ankle pitch 4/10 dùng hệ số 1,5 theo launcher v1.1.
5. Controller 50 Hz, physics 500 Hz. PD torque cập nhật mỗi physics step, giới hạn theo actuator XML. Quaternion WXYZ; orientation 6D flatten hai cột đầu theo từng hàng như C++.
6. Camera ngoài theo root, video ghi rõ SONIC pretrained/reference tracking. Telemetry có height, tilt, joint RMSE, state và tổng contact; tổng contact không được gọi là foot-contact metric.

Report chứa hash model/geometry/reference, provider ONNX, latency và tiêu chí ngã. Tiêu chí smoke định trước: pelvis dưới 0,35 m hoặc tilt trên 75°. Hoàn tất một clip không chứng minh balance khi gắp vật, mang tải, bị đẩy hoặc sim-to-real. Lượt local đã kiểm chứng ở [`reports/fullbody_sonic_smoke.json`](../reports/fullbody_sonic_smoke.json).

## VLA full-body cho gắp vật

Luồng đúng: RGB + ngôn ngữ + G1 proprioception → VLA dự đoán SONIC token 64D và hand targets → SONIC phối hợp toàn thân. Không ghép OSC action Panda 7D vào chân G1 hoặc suy ra balance loss từ data không có thân/chân. [Workflow NVIDIA](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html), [inference SONIC/VLA](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html).

Project hiện chưa có paired G1 task demonstrations/checkpoint VLA G1 đã fine-tune. Cần thu thập hoặc sinh demonstration tương tác vật trên scene G1 phù hợp, ghi RGB/state/tokens/hand targets cùng controller identity, rồi train trên GPU phù hợp. Clip locomotion không thay thế dữ liệu đó. Runtime demo từ chối `--vla-checkpoint` Panda và chỉ dẫn dùng stack VLA G1 chính thức.

`validate-g1` kiểm tra action 64+7+7, state lengths đúng G1, slice không âm/không chồng, controller digest, 50 Hz và frame hữu hạn. Đây là kiểm tra schema; replay/visual review và camera synchronization vẫn cần kiểm tra trên dữ liệu thật.

```bash
python -m robot_foundry.cli validate-g1 --dataset data/g1_sonic/cleaned_train
python -m robot_foundry.cli train-groot --config configs/groot_g1.json --dry-run
```

GR00T training dùng profile Linux/GPU lớn đã có; notebook T4 ×2 hiện vẫn là Panda. Xem [`DESIGN.md`](DESIGN.md), [`BENCHMARK.md`](BENCHMARK.md).
