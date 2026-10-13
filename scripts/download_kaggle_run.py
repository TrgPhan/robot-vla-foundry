"""Download one saved policy/run from Kaggle without copying its Python environments."""

import argparse
import hashlib
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path, PurePosixPath


def download_run(kernel, run_name, destination):
    from huggingface_hub import snapshot_download
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiDownloadKernelOutputRequest,
        ApiGetKernelRequest,
    )

    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel):
        raise ValueError("Use a kernel reference in owner/slug form")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("Invalid run name")
    owner, slug = kernel.split("/")
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 4 * 2**30:
        raise OSError(28, "Reserve at least 4 GiB for the downloaded policy and tokenizer assets")
    api = KaggleApi()
    api.authenticate()
    manifest_path = root / "download_manifest.json"
    with api.build_kaggle_client() as client:
        get = ApiGetKernelRequest()
        get.user_name, get.kernel_slug = owner, slug
        version = client.kernels.kernels_api_client.get_kernel(get).metadata.current_version_number
        manifest = {"kernel": kernel, "version": version, "run": run_name, "files": {}}
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(previous[k] != manifest[k] for k in ["kernel", "version", "run"]):
                raise ValueError("Destination belongs to a different Kaggle version; use a fresh directory")
            manifest = previous

        def publish():
            temp = manifest_path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(temp, manifest_path)

        def fetch(relative):
            name = PurePosixPath(relative)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Artifact paths must be relative to the saved run")
            path = root.joinpath(*name.parts)
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError("Artifact must remain inside destination")
            if relative in manifest["files"] and path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(4 * 2**20), b""):
                        digest.update(block)
                if digest.hexdigest() == manifest["files"][relative]["sha256"]:
                    print(f"Verified existing: {relative}", flush=True)
                    return
            request = ApiDownloadKernelOutputRequest()
            request.owner_slug, request.kernel_slug = owner, slug
            request.version_number = version
            request.file_path = f"robot-vla-foundry/outputs/{run_name}/{relative}"
            redirect = client.kernels.kernels_api_client.download_kernel_output(request)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".part")
            digest, size, next_progress = hashlib.sha256(), 0, 128 * 2**20
            print(f"Downloading v{version}: {relative}", flush=True)
            try:
                with urllib.request.urlopen(redirect.url, timeout=60) as response, temp.open("wb") as stream:
                    expected = int(response.headers.get("Content-Length", "0"))
                    while block := response.read(4 * 2**20):
                        stream.write(block)
                        digest.update(block)
                        size += len(block)
                        if size >= next_progress:
                            print(f"  {relative}: {size / 2**30:.2f} GiB", flush=True)
                            next_progress += 128 * 2**20
                if expected and size != expected:
                    raise OSError("Incomplete artifact transfer")
                os.replace(temp, path)
            except BaseException:
                temp.unlink(missing_ok=True)
                raise
            manifest["files"][relative] = {"bytes": size, "sha256": digest.hexdigest()}
            publish()

        publish()
        for name in [
            "run_config.json",
            "report.json",
            "git_provenance.json",
            "metrics.jsonl",
            "best/config.json",
            "best/stats.json",
            "best/data_manifest.json",
            "best/export_complete.json",
            "best/policy_preprocessor.json",
            "best/policy_postprocessor.json",
            "best/vlm_assets/foundry_revision.json",
            "best/model.safetensors",
        ]:
            fetch(name)
        # Processor normalization buffers are stored beside their JSON configurations.
        for config in ["policy_preprocessor.json", "policy_postprocessor.json"]:

            def buffers(value):
                if isinstance(value, dict):
                    return [name for item in value.values() for name in buffers(item)]
                if isinstance(value, list):
                    return [name for item in value for name in buffers(item)]
                return [value] if isinstance(value, str) and value.endswith(".safetensors") else []

            for name in buffers(json.loads((root / "best" / config).read_text(encoding="utf-8"))):
                fetch("best/" + name)
        assets = json.loads((root / "best/vlm_assets/foundry_revision.json").read_text(encoding="utf-8"))
        registry = json.loads(Path("configs/datasets.json").read_text(encoding="utf-8"))
        if assets != registry["smolvlm"]:
            raise ValueError("Tokenizer asset revision differs from the pinned recipe")
        snapshot_download(
            assets["repo_id"],
            revision=assets["revision"],
            local_dir=str(root / "best/vlm_assets"),
            ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.onnx"],
        )
        manifest["tokenizer_assets"] = {"repo_id": assets["repo_id"], "revision": assets["revision"]}
        manifest["scope"] = "Saved policy, processors and run reports; optimizer/prepared data not included"
        publish()
    print(f"Downloaded saved policy: {root / 'best'}", flush=True)
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("--run", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    download_run(args.kernel, args.run, args.destination)


if __name__ == "__main__":
    main()
