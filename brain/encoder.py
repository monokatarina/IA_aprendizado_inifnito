"""
Encoder de Percepção do Mundo
==============================
Z_t = f_enc(x_t) = Projection( Embedding_Llama(x_t) )

Projeta o embedding bruto do Llama (d_llama dimensional) para o
espaço compacto de representação do mundo (d_e = 256).
"""

import torch
import torch.nn as nn
import numpy as np

from config import Config


class WorldEncoder(nn.Module):
    """
    Encoder de Percepção do Mundo.

    Entrada : embedding bruto do Llama ∈ ℝ^{d_llama}
    Saída   : Z_t ∈ ℝ^{d_e}

    Arquitetura: Linear → GELU → LayerNorm → Linear → LayerNorm
    """

    def __init__(self, cfg: Config, llama_embed_dim: int = None):
        super().__init__()
        in_dim = llama_embed_dim if llama_embed_dim is not None else cfg.llama_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, cfg.d_h),
            nn.GELU(),
            nn.LayerNorm(cfg.d_h),
            nn.Linear(cfg.d_h, cfg.d_e),
            nn.LayerNorm(cfg.d_e),
        )
        self.d_e = cfg.d_e

    def forward(self, raw_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            raw_embedding: [batch, d_llama] ou [d_llama]
        Returns:
            Z_t: [batch, d_e] ou [d_e]
        """
        return self.proj(raw_embedding)

    def encode_numpy(self, raw: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Converte numpy → tensor, codifica e retorna vetor [d_e]."""
        t = torch.from_numpy(raw).float().to(device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        with torch.no_grad():
            z = self.forward(t)
        return z.squeeze(0)
