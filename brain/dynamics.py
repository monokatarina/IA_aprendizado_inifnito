"""
Dinâmica do Estado Interno
============================
f_dyn(S_{t,k-1}, Z_t, a, R_t) → S_{t,k}

Entrada : concat([S, Z, a, R]) ∈ ℝ^{d_s + 3·d_e}
Saída   : novo estado S ∈ ℝ^{d_s}

SelfPredictor estima o próximo estado para:
    L_self = ‖S_{t+1} − Ŝ_{t+1}‖²
"""

import torch
import torch.nn as nn

from config import Config


class StateDynamics(nn.Module):
    """
    Dinâmica do Estado Interno — f_dyn.

    Usado em dois contextos:
      1. Loop de voz interna (K passos): S_{t,k} = f_dyn(S_{t,k-1}, Z_t, a_int, R_t)
      2. Atualização real do estado:     S_{t+1}  = f_dyn(S_t*,     Z_t, a_ext, R_t)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.d_s + 3 * cfg.d_e  # S + Z + a + R
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.d_h),
            nn.GELU(),
            nn.LayerNorm(cfg.d_h),
            nn.Linear(cfg.d_h, cfg.d_h),
            nn.GELU(),
            nn.Linear(cfg.d_h, cfg.d_s),
            nn.LayerNorm(cfg.d_s),
        )

    def forward(
        self,
        S: torch.Tensor,   # [d_s]
        Z: torch.Tensor,   # [d_e]
        a: torch.Tensor,   # [d_e]
        R: torch.Tensor,   # [d_e]
    ) -> torch.Tensor:
        """Retorna S_{t,k} ∈ ℝ^{d_s}."""
        x = torch.cat([S, Z, a, R], dim=-1)
        return self.net(x)


class SelfPredictor(nn.Module):
    """
    Auto-Preditor do Próximo Estado — f_pred.

    Além de prever Ŝ_{t+1}, estima incerteza epistêmica (escalares >= 0),
    permitindo introspecção: "estou confiante" vs "isto é só uma simulação fraca".
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.mem_proj = nn.Linear(cfg.d_e, cfg.d_s)
        in_dim = cfg.d_s + cfg.d_e + cfg.d_e + cfg.d_s

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, cfg.d_h),
            nn.LayerNorm(cfg.d_h),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.state_head = nn.Linear(cfg.d_h, cfg.d_s)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(cfg.d_h, 1),
            nn.Softplus(),
        )

    def predict_with_uncertainty(
        self,
        S_star: torch.Tensor,
        Z: torch.Tensor,
        a_ext: torch.Tensor,
        R: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (Ŝ_{t+1}, u_t), com u_t >= 0."""
        mem = self.mem_proj(R)
        x = torch.cat([S_star, Z, a_ext, mem], dim=-1)
        h = self.backbone(x)
        s_hat = self.state_head(h)
        u_t = self.uncertainty_head(h).squeeze()
        return s_hat, u_t

    def forward(
        self,
        S_star: torch.Tensor,
        Z: torch.Tensor,
        a_ext: torch.Tensor,
        R: torch.Tensor,
    ) -> torch.Tensor:
        s_hat, _ = self.predict_with_uncertainty(S_star, Z, a_ext, R)
        return s_hat
