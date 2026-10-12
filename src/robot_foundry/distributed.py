"""One process per GPU, primary-only writes, and error propagation between ranks."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


class Runtime:
    def __init__(self, cfg):
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        name = cfg.get("device", "cuda")
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        if name.startswith("cuda"):
            if self.local_rank >= torch.cuda.device_count():
                raise RuntimeError("Not enough visible GPUs for this torchrun rank.")
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device(name)
        self.distributed = self.world_size > 1
        if self.distributed:
            dist.init_process_group(
                backend="nccl" if self.device.type == "cuda" else "gloo",
                init_method=os.environ.get("ROBOT_FOUNDRY_INIT_METHOD", "env://"),
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(minutes=10),
            )

    @property
    def primary(self):
        return self.rank == 0

    def all_true(self, condition):
        flag = torch.tensor(int(condition), device=self.device)
        if self.distributed:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def sum(self, tensor):
        if self.distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def gather(self, value):
        if not self.distributed:
            return [value]
        values = [None] * self.world_size
        dist.all_gather_object(values, value)
        return values

    def primary_call(self, fn):
        if not self.distributed:
            return fn()
        payload = [None, None]
        if self.primary:
            try:
                payload[0] = fn()
            except Exception as error:
                payload[1] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(payload, src=0, device=self.device)
        if payload[1]:
            raise RuntimeError(f"Primary process failed: {payload[1]}")
        return payload[0]

    def close(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


class StepBatchSampler:
    """Stateless per-microbatch sampling: worker prefetch cannot change resume order.

    Ranks own disjoint index residues, so simultaneous batches never duplicate across GPUs.
    Sampling remains with replacement within each rank's shard, as in the previous trainer.
    """

    def __init__(self, length, batch_size, accumulation, steps, seed, rank=0, world_size=1, start=0):
        if length < world_size or batch_size < 1 or accumulation < 1:
            raise ValueError("Dataset must have at least one sample/rank and positive batch/accumulation.")
        self.length, self.batch_size, self.accumulation = length, batch_size, accumulation
        self.steps, self.seed, self.rank, self.world_size, self.start = steps, seed, rank, world_size, start

    def __len__(self):
        return max(0, self.steps - self.start) * self.accumulation

    def __iter__(self):
        shard_size = (self.length - 1 - self.rank) // self.world_size + 1
        for micro in range(self.start * self.accumulation, self.steps * self.accumulation):
            seed = (self.seed + 101 + 1000003 * micro + 15485863 * self.rank) % (2**63 - 1)
            generator = torch.Generator().manual_seed(seed)
            local = torch.randint(shard_size, (self.batch_size,), generator=generator)
            yield (local * self.world_size + self.rank).tolist()
