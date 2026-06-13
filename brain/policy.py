"""
Políticas Interna e Externa
=============================

Política Interna (Voz Interna):
    π_int(S_{t,k-1}, R_t) → a_int_{t,k} ∈ ℝ^{d_e}
    Usada no loop de raciocínio de K passos.

Política Externa:
    π_ext(S_t*, Z_t, R_t, ε_t) → a_t^ext ∈ ℝ^{d_e}
    ε_t ~ N(0, σ²I)   (ruído exploratório)

    L_policy = ‖π_ext(S_t*, Z_t, R_t) − Z_{t+1}^real‖²
"""

import torch
import torch.nn as nn
import math

from config import Config


class InternalPolicy(nn.Module):
    """
    Política Interna — π_int.

    Gera ações internas para o loop de raciocínio (K passos).
    Entrada : concat([S, R]) ∈ ℝ^{d_s + d_e}
    Saída   : a_int ∈ ℝ^{d_e}
    """

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.d_s + cfg.d_e
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.d_h),
            nn.GELU(),
            nn.LayerNorm(cfg.d_h),
            nn.Linear(cfg.d_h, cfg.d_e),
            nn.Tanh(),
        )

    def forward(self, S: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: [d_s] — estado interno atual
            R: [d_e] — contexto recuperado da memória
        Returns:
            a_int: [d_e]
        """
        x = torch.cat([S, R], dim=-1)
        return self.net(x)


class ExternalPolicy(nn.Module):
    """
    Política Externa — π_ext.

    Gera embedding da intenção de resposta com ruído exploratório.
    Entrada : concat([S*, Z, R, P]) ∈ ℝ^{d_s + 2·d_e + d_p}
    Saída   : a_ext ∈ ℝ^{d_e}
    """

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.d_s + 2 * cfg.d_e + cfg.d_p
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.d_h),
            nn.GELU(),
            nn.LayerNorm(cfg.d_h),
            nn.Linear(cfg.d_h, cfg.d_h),
            nn.GELU(),
            nn.Linear(cfg.d_h, cfg.d_e),
            nn.Tanh(),
        )
        self.noise_std = cfg.noise_std

    def forward(
        self,
        S_star: torch.Tensor,    # [d_s]
        Z: torch.Tensor,         # [d_e]
        R: torch.Tensor,         # [d_e]
        P: torch.Tensor,         # [d_p]
        training: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            training: se True, adiciona ruído exploratório ε_t ~ N(0, σ²I)
        Returns:
            a_ext: [d_e]
        """
        a, _, _ = self.sample_action(S_star, Z, R, P, training=training)
        return a

    def sample_action(
        self,
        S_star: torch.Tensor,
        Z: torch.Tensor,
        R: torch.Tensor,
        P: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna ação, log-probabilidade e média da política.

        A política externa é tratada como gaussiana diagonal com desvio fixo
        `noise_std`, o que permite usar sinal de policy gradient mesmo com
        geração de texto fora da backprop direta.
        """
        x = torch.cat([S_star, Z, R, P], dim=-1)
        mean = self.net(x)

        if training:
            noise = torch.randn_like(mean) * self.noise_std
            action = mean + noise
            var = max(self.noise_std ** 2, 1e-6)
            log_scale = math.log(2.0 * math.pi * var)
            log_prob = -0.5 * (((action - mean).pow(2) / var) + log_scale).sum()
        else:
            action = mean
            log_prob = torch.tensor(0.0, device=mean.device)

        return action, log_prob, mean
