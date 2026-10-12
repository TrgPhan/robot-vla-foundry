"""Detect actual per-device capacity; two small GPUs do not pool VRAM under DDP."""

import platform
import shutil
import subprocess
import sys


def probe():
    import torch

    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            with torch.cuda.device(i):
                free, total = torch.cuda.mem_get_info()
                bf16 = torch.cuda.is_bf16_supported()
                try:
                    torch.ones(2, device=f"cuda:{i}").sum().item()
                    kernel_error = None
                except RuntimeError as error:
                    kernel_error = str(error)
            devices.append(
                {
                    "id": i,
                    "name": prop.name,
                    "total_gib": total / 2**30,
                    "free_gib": free / 2**30,
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "bf16": bf16,
                    "kernel_error": kernel_error,
                }
            )
    driver = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        driver = result.stdout.strip()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver": driver,
        "gpus": devices,
        "compiled_cuda_arches": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }


def require_gpu(profile, num_gpus=1):
    report = probe()
    minimum = {"groot": 39.0, "smolvla": 14.0}[profile]
    if len(report["gpus"]) < num_gpus or any(g["total_gib"] < minimum for g in report["gpus"][:num_gpus]):
        raise RuntimeError(
            f"{profile} profile needs {num_gpus} GPU(s), each >= {minimum} GiB. "
            "Use tiny_bc on this machine or a larger GPU. DDP does not add VRAM together."
        )
    if any(g["kernel_error"] for g in report["gpus"][:num_gpus]):
        raise RuntimeError(
            "CUDA kernel probe failed. Install a wheel supporting this GPU/driver; see doctor."
        )
    if profile == "groot":
        if sys.version_info[:2] != (3, 12) or platform.system() != "Linux":
            raise RuntimeError(
                "Official GR00T profile targets Linux, Python 3.12. Use its isolated environment."
            )
        if any(not g["bf16"] for g in report["gpus"][:num_gpus]):
            raise RuntimeError(
                "GR00T profile requires BF16-capable GPUs; P100/T4 are unsuitable for this recipe."
            )
        if not report["torch_cuda"] or tuple(map(int, report["torch_cuda"].split(".")[:2])) < (12, 6):
            raise RuntimeError(
                "The reference GR00T profile needs its CUDA 12.6+ environment, not the Kaggle cu118 env."
            )
    return report
