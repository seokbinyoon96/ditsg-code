"""MAI block: masked multivariate attention -> agent attention -> FFN, adaLN-Zero."""

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def to_tokens(v, nf):
    """[B, N, D] -> [B, N*F, D]"""
    B, N, D = v.shape
    return v.unsqueeze(2).expand(-1, -1, nf, -1).reshape(B, N * nf, D)


def agent_mask(num_agents, nf, device):
    """[N*F, N*F] bool, True = blocked (attend only within the same agent)."""
    agent = torch.arange(num_agents * nf, device=device) // nf
    return agent[:, None] != agent[None, :]


class MAIDiTBlock(nn.Module):
    """x [B, N*F, D], cond [B, N, D] -> x [B, N*F, D]"""

    def __init__(self, d_model, head, d_ff, dropout, nf=3):
        super().__init__()
        self.d_model = d_model
        self.nf = nf

        self.inverted_attention = nn.MultiheadAttention(
            d_model, head, dropout=dropout, batch_first=True
        )
        self.agent_attention = nn.MultiheadAttention(
            d_model * nf, head, dropout=dropout, batch_first=True
        )
        self.ffn_in = nn.Linear(d_model, d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model * nf, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

        # (shift, scale, gate) x 3, zero-init
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 9 * d_model))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def _tok(self, v):
        return to_tokens(v, self.nf)

    def _agt(self, v):
        """[B, N, D] -> [B, N, F*D]"""
        return v.repeat(1, 1, self.nf)

    def forward(self, x, cond):
        B, L, D = x.shape
        N = L // self.nf
        sh1, sc1, g1, sh2, sc2, g2, sh3, sc3, g3 = self.modulation(cond).chunk(9, dim=-1)

        # masked multivariate attention
        h = modulate(self.norm1(x), self._tok(sh1), self._tok(sc1))
        h, feature_attn = self.inverted_attention(h, h, h, attn_mask=agent_mask(N, self.nf, x.device))
        x = x + self._tok(g1) * self.dropout(h)

        # agent attention
        xa = x.reshape(B, N, self.nf * D)
        h = modulate(self.norm2(xa), self._agt(sh2), self._agt(sc2))
        h, agent_attn = self.agent_attention(h, h, h)
        x = (xa + self._agt(g2) * self.dropout(h)).reshape(B, L, D)

        # feed forward
        h = modulate(self.norm3(x), self._tok(sh3), self._tok(sc3))
        h = self.ffn_out(self.act(self.ffn_in(h)))
        x = x + self._tok(g3) * self.dropout(h)

        return x, agent_attn, feature_attn
