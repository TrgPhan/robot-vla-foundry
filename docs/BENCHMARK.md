# Protocol đánh giá

## Manipulation

Benchmark chính là **LIBERO Object**: 10 task gắp các vật khác nhau vào giỏ. Spatial là benchmark phụ để kiểm tra vị trí tương đối. Đây là Panda manipulation, không đo cân bằng humanoid. [LIBERO chính thức](https://libero-project.github.io/main.html), [NVIDIA benchmark workflow](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md).

1. Dữ liệu: episode-disjoint train/val/test. Stats lấy từ train. Selection checkpoint theo validation normalized RMSE.
2. Open-loop: normalized RMSE, raw MAE từng chiều, first-action error, padding mask. Không gộp centimetre, radian, gripper thành một raw MAE vật lý có ý nghĩa.
3. Closed-loop: 50 **unique official initial states/task**, 300 policy steps cho Object, 10 no-op settling steps, execution horizon 4 mặc định. Report max steps/horizon, checkpoint SHA, seed, success của từng trial và macro success.
4. Run chính ít nhất 3 seed; không lặp state để tăng sample count trong một run. Wilson interval trong evaluator là mô tả; pooled interval không mô hình hóa tương quan giữa task/seed.
5. Chọn hyperparameter trên validation; chạy test và benchmark cuối khi đã cố định recipe. Nếu dùng các initial states/scene đã xuất hiện trong demonstration, đó là protocol benchmark chuẩn cùng scene, không được gọi là unseen-scene/OOD.

Local smoke đã chạy 1 state/task, 100 steps và **0/10**. Không thể so nó trực tiếp với kết quả paper dùng nhiều dữ liệu, nhiều GPU và horizon/protocol khác. TinyBC là baseline pipeline, không có language pretraining hay zero-shot generalization.

## TinyBC/SmolVLA simulation

```bash
<SIM_PY> -m robot_foundry.cli eval-libero \
  --checkpoint outputs/local_tiny_bc/best \
  --suite libero_object --episodes-per-task 50 \
  --max-steps 300 --execution-horizon 4 --seed 42 \
  --output outputs/libero_object_seed42
```

Thay `<SIM_PY>` bằng `.venv-sim/bin/python` trên Linux hoặc `.venv-sim-local/Scripts/python.exe` trên Windows. Output chứa `trials.jsonl` và `report.json`. Dùng thư mục mới cho mỗi run, evaluator không nối thêm một lượt đầy đủ vào file cũ.

SmolVLA cần hai terminal/môi trường:

```bash
<TRAIN_PY> scripts/policy_server.py --checkpoint outputs/kaggle_smolvla_full/best --backend smolvla --port 8765
<SIM_PY> -m robot_foundry.cli eval-libero --checkpoint outputs/kaggle_smolvla_full/best --backend smolvla --policy-url http://127.0.0.1:8765 --suite libero_object --episodes-per-task 50 --max-steps 300 --output outputs/libero_smolvla_seed42
```

Camera conversion quay 180° theo NVIDIA wrapper: `image[::-1, ::-1]`. State gồm EEF XYZ, axis-angle, hai gripper qpos. Dataset gripper 0=closed/1=open chuyển sang simulator +1=closed/-1=open. Unit tests kiểm tra dấu và tránh đổi input tại chỗ. Không dùng adapter này cho LeRobot v3 conversion có action convention khác mà chưa xác minh.

## GR00T native

Dùng server/client chính thức ở pinned repo. Server chạy trong `vendor/groot/.venv`; LIBERO client chạy trong environment benchmark chính thức. Setup tham khảo `vendor/groot/gr00t/eval/sim/LIBERO/setup_libero.sh`; script upstream này xóa config `~/.libero`, nên project dùng `scripts/setup_sim.py` cho evaluator riêng và không tự chạy script đó trên máy bạn.

Ví dụ start server ở checkout:

```bash
python gr00t/eval/run_gr00t_server.py --model-path /ABS/CHECKPOINT --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper
```

Evaluator native `gr00t/eval/rollout_policy.py` là client cho từng environment đã đăng ký, khác CLI riêng của project. Dùng task ID `libero_sim/pick_up_the_milk_and_place_it_in_the_basket`, `--n-episodes 50`, `--max-episode-steps 300`, `--n-action-steps 4`, host `127.0.0.1` và port server `5555`. Repeat cho đủ 10 task và lưu report mỗi task. Không dùng checkpoint TinyBC trong server GR00T.

Muốn open-loop holdout native trước rollout, dùng `gr00t/eval/open_loop_eval.py --help` của checkout đã khóa và `data/native/libero/<source>/val`. File modality/stats đã export từ training set. GR00T recipe này chưa có số đo rollout local vì GPU không đủ.

## Whole-body G1 khi có môi trường

SONIC motion tracking: success không early termination; local/global MPJPE; joint tracking, contact/foot slip, base tilt và action saturation. `scripts/sonic_plan.py` tạo lệnh eval chính thức trên holdout motion.

Task humanoid: báo cáo cả **manipulation success và fall-free success**; time-to-completion, base roll/pitch p95/max, foot slip, wrist error, joint/torque-limit events, dropped object và payload. Define fall/contact thresholds và simulator robot/object model trước test. Thử standing reach trước, sau đó payload và perturbations. MPJPE tốt trên mocap không chứng minh gắp vật tốt hoặc sim-to-real an toàn.

Project hiện chưa tạo benchmark object-interaction G1/Isaac Lab riêng, chưa có dữ liệu train G1 và chưa đo các metric này. Các cấu hình SONIC/G1 là đường tích hợp với upstream, cần dữ liệu và tài nguyên phù hợp.
