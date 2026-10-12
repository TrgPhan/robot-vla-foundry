"""Small CNN + language-token + state action-chunk BC baseline; no foundation-model claim."""

import torch
from torch import nn


class TinyChunkPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, num_cameras, horizon, width=128):
        super().__init__()
        self.spec = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "num_cameras": num_cameras,
            "horizon": horizon,
            "width": width,
        }
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, width, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.camera_embed = nn.Parameter(torch.randn(num_cameras, 1, width) * 0.02)
        self.position_embed = nn.Parameter(torch.randn(1, 4, width) * 0.02)
        self.text = nn.Embedding(2048, width, padding_idx=0)
        self.state = nn.Sequential(nn.Linear(state_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.queries = nn.Parameter(torch.randn(horizon, width) * 0.02)
        layer = nn.TransformerDecoderLayer(
            width, 4, width * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, action_dim))

    def forward(self, batch):
        images = batch["images"].float() / 255
        b, v, c, h, w = images.shape
        if self.training:
            brightness = torch.empty(b, 1, 1, 1, 1, device=images.device).uniform_(0.9, 1.1)
            images = (images * brightness).clamp(0, 1)
        vision = self.vision(images.reshape(b * v, c, h, w)).flatten(2).transpose(1, 2)
        vision = vision.reshape(b, v, 4, -1) + self.camera_embed + self.position_embed
        token_ids = batch["tokens"]
        token_mask = (token_ids != 0).unsqueeze(-1)
        text = (self.text(token_ids) * token_mask).sum(1) / token_mask.sum(1).clamp_min(1)
        memory = torch.cat(
            [vision.flatten(1, 2), text.unsqueeze(1), self.state(batch["state"]).unsqueeze(1)], dim=1
        )
        return self.head(self.decoder(self.queries.unsqueeze(0).expand(b, -1, -1), memory))
