"""Time embeddings for t (flow time) and tau (remaining time)."""

import math

import torch
import torch.nn as nn


def sinusoidal_time_encoding(tau, dim):
    """[...] -> [..., dim]"""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=tau.dtype) / half
    )
    args = tau.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def time_mlp(d_model):
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.SiLU(),
        nn.Linear(d_model, d_model),
    )
