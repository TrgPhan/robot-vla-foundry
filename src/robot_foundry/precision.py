"""T4-compatible FP16 autocast; weights and loss reductions remain FP32."""

from contextlib import nullcontext

import torch


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision not in ["fp16", "bf16"]:
        raise ValueError(f"Unknown precision: {precision}")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("This FP16 profile requires CUDA; use fp32 for CPU tests.")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This GPU does not support BF16; T4 uses FP16.")
    return torch.autocast(
        device_type=device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16
    )
