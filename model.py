"""DiTSG: multi-agent diffusion transformer (flow matching, x-prediction)."""

import torch.nn as nn

import config
from block import MAIDiTBlock, modulate, to_tokens
from embedding import sinusoidal_time_encoding, time_mlp


class DiTSG(nn.Module):
    """z [B, T, N*F], t [B], tau [B, N] -> x_pred [B, T, N*F]"""

    def __init__(self, d_model, head, d_ff, seq_len, num_layers, dropout,
                 nf=3, time_emb_scale=1000.0):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.nf = nf
        self.time_emb_scale = time_emb_scale

        self.embedding = nn.Linear(seq_len, d_model)    # one token per (agent, feature)
        self.t_embed = time_mlp(d_model)
        self.tau_embed = time_mlp(d_model)

        self.blocks = nn.ModuleList([
            MAIDiTBlock(d_model, head, d_ff, dropout, nf) for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.modulation_out = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.modulation_out[-1].weight)
        nn.init.zeros_(self.modulation_out[-1].bias)

        self.head_out = nn.Linear(d_model, seq_len)
        nn.init.zeros_(self.head_out.weight)
        nn.init.zeros_(self.head_out.bias)

    def condition(self, t, tau):
        """t [B], tau [B, N] -> cond [B, N, D]"""
        s = self.time_emb_scale
        t_emb = self.t_embed(sinusoidal_time_encoding(t * s, self.d_model))
        tau_emb = self.tau_embed(sinusoidal_time_encoding(tau * s, self.d_model))
        return t_emb.unsqueeze(1) + tau_emb

    def forward(self, z, t, tau, return_attention=False):
        if t.dim() == 0:
            t = t.expand(z.size(0))

        x = self.embedding(z.transpose(1, 2))       # [B, N*F, D]
        cond = self.condition(t, tau)               # [B, N, D]

        attentions = []
        for block in self.blocks:
            x, agent_attn, feature_attn = block(x, cond)
            attentions.append((agent_attn, feature_attn))

        shift, scale = self.modulation_out(cond).chunk(2, dim=-1)
        x = modulate(self.norm_out(x), to_tokens(shift, self.nf), to_tokens(scale, self.nf))

        x_pred = self.head_out(x).transpose(1, 2)   # [B, T, N*F]
        return (x_pred, attentions) if return_attention else x_pred


def build_model(seq_len, device=None, verbose=True):
    device = config.DEVICE if device is None else device
    model = DiTSG(
        d_model=config.D_MODEL,
        head=config.HEAD,
        d_ff=config.D_FF,
        seq_len=seq_len,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        nf=config.NF,
        time_emb_scale=config.TIME_EMB_SCALE,
    ).to(device)
    if verbose:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"DiTSG parameters: {n:,} | {device}")
    return model
