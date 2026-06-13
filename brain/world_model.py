"""
Modelo do Mundo
=================
f_world(Z_t, a_t^ext) → Ẑ_{t+1}

Prediz o próximo estado perceptual dado o estado atual e a ação tomada.
Usado para recompensa de curiosidade (surpresa) e imaginação (rollout).

    L_world = ‖Ẑ_{t+1} − Z_{t+1}^real‖²

Bônus de Imaginação (horizonte H):
    reward_imag = Σ_{h=1}^{H} −‖Ẑ_{t+h} − Z_{t+h−1}‖²
"""

import torch
import torch.nn as nn

from config import Config


class WorldModel(nn.Module):
    """Modelo do Mundo — f_world."""

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.d_e + cfg.d_e  # Z + a_ext
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.d_h),
            nn.GELU(),
            nn.LayerNorm(cfg.d_h),
            nn.Linear(cfg.d_h, cfg.d_h),
            nn.GELU(),
            nn.Linear(cfg.d_h, cfg.d_e),
            nn.LayerNorm(cfg.d_e),
        )

    def forward(self, Z: torch.Tensor, a_ext: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Z    : [d_e] — embedding atual do mundo
            a_ext: [d_e] — embedding da ação externa
        Returns:
            Z_hat_next: [d_e] — próximo estado previsto
        """
        x = torch.cat([Z, a_ext], dim=-1)
        return self.net(x)

    def rollout(
        self,
        Z: torch.Tensor,
        a_ext: torch.Tensor,
        H: int = 1,
    ) -> torch.Tensor:
        """
        Imaginação em H passos.

        reward_imag = Σ_{h=1}^{H} −‖Ẑ_{t+h} − Z_{t+h−1}‖²

        Returns:
            reward_imag: tensor escalar (diferenciável)
        """
        reward = torch.tensor(0.0, device=Z.device)
        Z_curr = Z.detach().clone()
        a_curr = a_ext

        for _ in range(H):
            Z_next  = self.forward(Z_curr, a_curr)
            reward -= torch.norm(Z_next - Z_curr, p=2).pow(2)
            Z_curr  = Z_next.detach()

        return reward
