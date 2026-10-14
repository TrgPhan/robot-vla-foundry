# Dataset catalogue và thu thập thực tế

Dataset/model revisions và upstream commits đã khóa trong `configs/datasets.json` và `configs/upstream_lock.json`. `reports/source_audit.json` xác nhận các revision truy cập được qua Hub ngày 12/10/2026. Token đọc từ môi trường/Hugging Face cache; không ghi token vào source.

Notebook G1 chính mới **sinh paired simulation data** từ [walking_quip](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428) và [macarena](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic_deploy/reference/example/macarena_001__A545). SONIC teacher chạy MuJoCo và ghi current RGB/state70/token64; không chuyển dữ liệu Panda thành G1. Nguồn, split, raw LeRobot-style export và phạm vi motion-only được mô tả tại [`G1_TRAINING.md`](G1_TRAINING.md).

| Nguồn | Vai trò | Trạng thái trong project |
|---|---|---|
| [LeRobot LIBERO](https://huggingface.co/datasets/lerobot/libero) | Dataset công khai LeRobot v3, Panda, 40 task | Khảo sát; FPS/conversion khác IPEC, không trộn mặc định |
| [IPEC LIBERO Object](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot) | Nguồn train chính, public LeRobot v2.1 | Đã tải và xử lý 50 episode |
| [IPEC LIBERO Spatial](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot) | Bổ sung cùng Panda/action convention | Đã tải và xử lý 50 episode |
| [SO100 pickplace](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) | Manipulation thật, nhánh SO100 riêng | Khảo sát metadata |
| [SO100 sorting](https://huggingface.co/datasets/lerobot/svla_so100_sorting), [stacking](https://huggingface.co/datasets/lerobot/svla_so100_stacking) | Bổ sung nhánh SO100 | Khảo sát metadata; không dùng trong train Panda |
| [NVIDIA PhysicalAI X-Embodiment Sim](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | GR1, bimanual, kitchen simulation | Khảo sát; chuyển embodiment riêng, không phải G1 token data |
| [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) | Motion prior, có G1 retargets | Gated; chưa tải motion corpus, không có paired robot RGB/grasp labels |
| [GEAR-SONIC weights](https://huggingface.co/nvidia/GEAR-SONIC), [public sample motions](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/b042411fae38ee4d1af9aac82a37a1f8d14d6dd0/gear_sonic_deploy/reference/example) | Teacher/controller cho paired G1 motion simulation | Đã tải encoder/decoder, walking và macarena; đã sinh và kiểm tra paired data |

BONES-SEED có 142.220 chuyển động và representation G1, nhưng repository cần đồng ý điều khoản truy cập. Nó bổ sung motion tracking, không biến video thao tác Panda thành whole-body G1 demonstrations. [Dataset card của tác giả](https://huggingface.co/datasets/bones-studio/seed).

Public format và public access không đồng nghĩa mọi nguồn có cùng license. Registry ghi license theo card; IPEC conversions chưa khai báo trường này. LeRobot datasets nêu Apache-2.0; NVIDIA simulation nêu CC-BY-4.0; SONIC weights dùng điều khoản NVIDIA trên model card. Không redistributing raw data trong source zip.

## Chọn mixture

Nhánh **Panda** mặc định chỉ Object + Spatial: cùng Panda, cùng thứ tự 8D state/7D OSC action, hai camera và 20 Hz. Downloader kiểm tra robot/FPS/features/cameras. Recipe yêu cầu `action_convention` rõ ràng; matching vector size không đủ. Nhánh **G1** dùng dataset riêng từ collector, cùng controller identity và action64, không trộn với Panda/SO100/GR1.

Nếu chạy full Kaggle, đặt `episodes_per_task=null`. Nguồn Object có 454 episode và Spatial có 432 episode ở revision khảo sát; chạy cap nhỏ trước để kiểm tra disk/time. Chỉ các file episode được chọn và metadata được tải. Một file video v3 có thể chứa nhiều episode nên subset download v3 vẫn tải file chứa episode đó.

SO100 camera/joint ordering tương thích về schema giữa ba nguồn, nhưng cần kiểm tra calibration, units và scene distribution trước mixture thực. Tạo recipe riêng với `so100_absolute_joint_driver_units_calibration_required`; không gọi evaluator LIBERO cho checkpoint SO100.

## Chất lượng và provenance

Kiểm tra NaN/Inf, dimensions, frame index, FPS, độ dài episode, video PTS/coverage và action padding. Split whole episodes stratified theo task; mean/std chỉ từ train full frames. Mỗi prepared episode chứa hash file nguồn và manifest ghi revision/episode ID/cache digest.

Split hiện tại ở cấp episode. Nếu source có nhiều episode gần trùng nhau, hoặc session/object IDs có sẵn, chuyển sang split nhóm session/object để đánh giá tổng quát hóa chặt hơn. Recipe hiện chưa có thông tin session đủ để chứng minh session-disjoint.
