# Validation record — 12/10/2026

- RTX A1000 Laptop 4 GB, Windows, Python 3.12.10, PyTorch 2.7.1/cu118.
- Real LIBERO v2 datasets: 100 selected episodes; train-only stats; RGB decoded and aligned; 60/20/20 split.
- TinyBC: 600 optimizer steps, test normalized RMSE 0.772789, mean-action baseline 0.989422.
- LIBERO environment created and ran all 10 Object tasks: 1 initial state each, 100 steps. Success 0/10. This is a runtime smoke test, not a full benchmark.
- Real pretrained SmolVLA: one CPU optimizer step with a tail-padding example; finite loss/gradients; checkpoint export/reload; finite `[1,16,7]` prediction. See `smolvla_smoke.json`.
- CPU resume regression: an interrupted optimizer run produces exactly the same parameters and sampling state as an uninterrupted run. GPU runs use deterministic cuDNN but are not claimed bitwise reproducible across GPU types.
- HTTP transport passed a real request through a running local TinyBC policy server. PNG camera roundtrip is exact; remote policy metadata is checked.
- LeRobot v3 SO100 metadata and video offset paths inspected; full v3 video training was not run.
- Native GR00T export: actual selected raw videos transcoded to H.264, data reindexed, train-only per-source stats propagated to val/test. CLI flags matched the pinned NVIDIA dataclass. Actual training was rejected by the intended VRAM check on this GPU.
- SONIC pinned repo bootstrapped; motion evaluation plan generated. Isaac Lab/SONIC dynamics were not executed on this Windows machine.
- Kaggle Notebook code cells compiled. Linux/Python 3.12 dependency graph for the pinned LeRobot SmolVLA profile resolved with CUDA 11.8 wheel constraints. The complete Notebook has not been executed on Kaggle.
- Git release notebook clones the public repository and delegates training to `scripts/run_kaggle.py`. Each experiment keeps its own prepared manifest, run config, source commit and checkpoint. Full remote Kaggle execution is still unmeasured.

Full VLA convergence, benchmark success, humanoid dynamics and sim-to-real performance remain unmeasured.

T4 ×2 release (`kaggle-v2`):

- Actual two-process Gloo/DDP training passed locally: both ranks have identical final parameters; five validation samples are counted once despite uneven sharding; interruption/resume reproduces the uninterrupted weights exactly with worker prefetch enabled.
- Actual pretrained SmolVLA trained two steps on two CPU processes with unused-parameter scanning disabled. Weight digests match both before best-checkpoint loading and after loading. See `smolvla_ddp_cpu_smoke.json`.
- Actual pretrained SmolVLA FP16 trained one optimizer step on RTX A1000: finite validation/test predictions, zero AMP skips, peak allocation 3143.33 MiB. This diagnostic used 96px/H16 and explicitly bypassed the production >=14 GiB guard within the temporary test process; it does not measure the production 256px/H50 recipe. See `smolvla_fp16_local_smoke.json`.
- FP16 autocast, gradient accumulation, fused AdamW and GradScaler checkpoint passed on the single local CUDA GPU using TinyBC. This does not measure SmolVLA memory on T4.
- Injected gradient overflow is skipped, scale halves, and subsequent optimizer updates stay finite. Distributed evaluation agrees with single-process sufficient statistics within 1e-8.
- Lossless packed cache returns exactly the same RGB, state, action, padding mask and language tokens as the original NPZ reader.
- One/two-GPU full and benchmark recipes retain effective batch 16. Explicit two-GPU selection fails when only one GPU is present.
- Notebook cells compile; production multi-GPU launcher uses Linux `torchrun`/NCCL. Actual T4 ×2 throughput, full SmolVLA convergence and full Kaggle notebook execution remain unmeasured.
