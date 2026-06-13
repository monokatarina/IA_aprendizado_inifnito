"""
Filtro de Aprendizado (Learning Gate)
=======================================
relevance_t = (1/|M|) · Σ sim(Z_t, Z_m)
g_t = σ( f_gate(relevance_t, r_t) )

g_t ∈ [0,1] controla quanto a experiência atual deve influenciar
o gradiente de aprendizado.

    L_total = g_t · ( λ_w·L_world + λ_s·L_self + λ_p·L_policy + λ_c·L_critic − η·r_t )
"""

import torch
import torch.nn as nn

from config import Config


class LearningGate(nn.Module):
    """Gate de Aprendizado — g_t = σ(f_gate(relevance_t, r_t))."""

    def __init__(self, cfg: Config):
        super().__init__()
        # Entrada: [relevance, r_t] — dois escalares
        self.gate = nn.Sequential(
            nn.Linear(2, cfg.d_h // 4),
            nn.GELU(),
            nn.Linear(cfg.d_h // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, relevance: float, r_t: float) -> torch.Tensor:
        """
        Args:
            relevance: relevância média com a memória (escalar Python)
            r_t      : recompensa intrínseca (escalar Python)
        Returns:
            g_t: tensor escalar em [0, 1]
        """
        device = next(self.parameters()).device
        x = torch.tensor([relevance, r_t], dtype=torch.float32, device=device)
        return self.gate(x).squeeze()
