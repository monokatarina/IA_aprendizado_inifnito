"""
Crítico Interno
================
V_t = Critic(S_t*, Z_t, R_t)

Avalia o valor do estado atual. O target é o negativo da surpresa:
    V̂_t = −surprise_t = −‖Z_{t+1} − Ẑ_{t+1}‖₂

    L_critic = (V_t − V̂_t)²
"""

import torch
import torch.nn as nn

from config import Config


class Critic(nn.Module):
    """Crítico Interno — avalia o valor do estado considerando personalidade."""

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.d_s + 2 * cfg.d_e + cfg.d_p  # S* + Z + R + P
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.d_h),
            nn.GELU(),
            nn.LayerNorm(cfg.d_h),
            nn.Linear(cfg.d_h, cfg.d_h // 2),
            nn.GELU(),
            nn.Linear(cfg.d_h // 2, 1),
        )

    def forward(
        self,
        S_star: torch.Tensor,  # [d_s]
        Z: torch.Tensor,       # [d_e]
        R: torch.Tensor,       # [d_e]
        P: torch.Tensor,       # [d_p] — personalidade latente
    ) -> torch.Tensor:
        """Retorna V_t como tensor escalar, condicionado pela personalidade."""
        x = torch.cat([S_star, Z, R, P], dim=-1)
        return self.net(x).squeeze(-1)
