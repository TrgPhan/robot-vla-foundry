"""Bounded episode cache and mask-aware metrics shared by both training backends."""

import bisect
import hashlib
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import read_json


def tokenize(prompt, length=40):
    words = re.findall(r"[a-z0-9]+", prompt.lower())[:length]
    tokens = np.zeros(length, dtype=np.int64)
    for i, word in enumerate(words):
        tokens[i] = 1 + int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "little") % 2047
    return tokens


class EpisodesDataset(Dataset):
    def __init__(self, root, split, normalize=True, cache_size=2):
        self.root = Path(root)
        self.manifest = read_json(self.root / "manifest.json")
        self.records = [r for r in self.manifest["episodes"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"Empty {split} split.")
        self.ends = np.cumsum([r["samples"] for r in self.records]).tolist()
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = OrderedDict()
        stats = read_json(self.root / "stats.json")
        self.state_mean = np.asarray(stats["observation.state"]["mean"], np.float32)
        self.state_std = np.asarray(stats["observation.state"]["std"], np.float32)
        self.action_mean = np.asarray(stats["action"]["mean"], np.float32)
        self.action_std = np.asarray(stats["action"]["std"], np.float32)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_idx = bisect.bisect_right(self.ends, index)
        row = index - (self.ends[episode_idx - 1] if episode_idx else 0)
        if episode_idx not in self.cache:
            with np.load(self.root / self.records[episode_idx]["path"], allow_pickle=False) as a:
                self.cache[episode_idx] = {
                    k: a[k] for k in ["images", "states", "actions", "valid", "prompts"]
                }
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode_idx)
        a = self.cache[episode_idx]
        state, action = a["states"][row], a["actions"][row]
        if self.normalize:
            state = (state - self.state_mean) / self.state_std
            action = (action - self.action_mean) / self.action_std
        prompt = str(a["prompts"][row])
        return {
            "images": torch.from_numpy(a["images"][row].copy()),
            "state": torch.from_numpy(state.copy()),
            "action": torch.from_numpy(action.copy()),
            "valid": torch.from_numpy(a["valid"][row].copy()),
            "tokens": torch.from_numpy(tokenize(prompt)),
            "prompt": prompt,
        }


def masked_loss(pred, target, valid, velocity_weight=0.05):
    mask = valid.unsqueeze(-1).expand_as(target)
    denom = mask.sum().clamp_min(1)
    loss = (torch.nn.functional.smooth_l1_loss(pred, target, reduction="none") * mask).sum() / denom
    if velocity_weight and pred.shape[1] > 1:
        pair_mask = (valid[:, 1:] & valid[:, :-1]).unsqueeze(-1).expand_as(target[:, 1:])
        error = torch.nn.functional.smooth_l1_loss(
            pred[:, 1:] - pred[:, :-1], target[:, 1:] - target[:, :-1], reduction="none"
        )
        loss = loss + velocity_weight * (error * pair_mask).sum() / pair_mask.sum().clamp_min(1)
    return loss
