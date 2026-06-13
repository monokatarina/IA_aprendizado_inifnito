"""
Personalidade emergente da IA.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from config import Config


class Personality(nn.Module):
    """Modelo de personalidade latente P com atualização por valência."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.d_p = cfg.d_p
        self.valence_dim = cfg.valence_dim

        # Vetor latente persistente que também recebe gradiente do fluxo principal.
        self.P = nn.Parameter(torch.zeros(cfg.d_p))

        # Mapeia valência emocional observada para direção de ajuste em P.
        self.valence_proj = nn.Linear(cfg.valence_dim, cfg.d_p)

        # Histórico para diagnóstico: (step, P_snapshot, valence_snapshot).
        self.history: List[Tuple[int, torch.Tensor, torch.Tensor]] = []

    def forward(self, valence: torch.Tensor | None = None) -> torch.Tensor:
        return self.P

    def update_with_valence(self, valence: torch.Tensor, lr: float = 1e-4, step: int = 0):
        """
        Registra valência para diagnóstico e histórico.
        P é atualizado apenas via backprop do RL, não via atualização heurística.
        """
        if valence.ndim != 1:
            valence = valence.view(-1)
        if valence.shape[0] != self.valence_dim:
            return

        # Registra snapshot de P e valência para análise posterior.
        self.history.append(
            (
                int(step),
                self.P.detach().cpu().clone(),
                valence.detach().cpu().clone(),
            )
        )
        if len(self.history) > 200:
            self.history = self.history[-200:]

    def consolidate(self, memory_text: str, llm_bridge):
        """Extrai traços narrativos de personalidade a partir de memória relevante."""
        prompt = (
            "Com base nas minhas experiências, escreva 3 traços de personalidade "
            "que estou desenvolvendo (ex: cética, curiosa, assertiva). "
            "Responda APENAS os traços separados por vírgula.\n"
            f"Memória relevante: {memory_text[:500]}"
        )
        try:
            response = llm_bridge.generate(
                user_message=prompt,
                system_prompt="Você é EcoMental em introspecção.",
                history=[],
                max_tokens=50,
            )
            _ = [t.strip() for t in response.split(",")[:3]]
        except Exception:
            pass
