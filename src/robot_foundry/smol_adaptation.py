"""Reinitialize embodiment boundary projections while retaining the pretrained VLM/action expert."""

import math

import torch


def adapt_g1_boundaries(policy, state_dim=70, action_dim=64):
    core, config = policy.model, policy.config
    state_capacity, action_capacity = math.ceil(state_dim / 32) * 32, math.ceil(action_dim / 32) * 32
    original = {"max_state_dim": config.max_state_dim, "max_action_dim": config.max_action_dim}
    replacements = {
        "state_proj": (state_capacity, core.state_proj.out_features),
        "action_in_proj": (action_capacity, core.action_in_proj.out_features),
        "action_out_proj": (core.action_out_proj.in_features, action_capacity),
    }
    for name, (inputs, outputs) in replacements.items():
        old = getattr(core, name)
        setattr(
            core,
            name,
            torch.nn.Linear(
                inputs, outputs, bias=old.bias is not None, device=old.weight.device, dtype=old.weight.dtype
            ),
        )
    config.max_state_dim, config.max_action_dim = state_capacity, action_capacity
    return {
        "method": "fresh G1 state/action boundary projections; retain pretrained VLM and expert",
        "source_dimensions": original,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_state_dim": state_capacity,
        "max_action_dim": action_capacity,
        "reinitialized_layers": list(replacements),
    }
