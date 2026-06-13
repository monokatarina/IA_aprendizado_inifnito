"""
Recompensas Intrínsecas
========================
r_t = α·novelty_t + β·progress_t + γ·surprise_t − λ_c·V_t

| Componente  | Fórmula                              | Descrição                    |
|-------------|--------------------------------------|------------------------------|
| novelty_t   | ‖Z_{t+1} − Z_t‖₂                   | Quão diferente é o novo estado|
| progress_t  | |L_world_{t-1} − L_world_t|         | Redução do erro de mundo      |
| surprise_t  | ‖Z_{t+1} − Ẑ_{t+1}‖₂              | Erro de previsão do mundo     |
| V_t         | valor do crítico (penalidade)        | Penalidade por alta estimativa|
"""

import torch
from config import Config


class IntrinsicRewards:
    """Calculador de Recompensas Intrínsecas."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._prev_world_loss: float = 0.0      # rastreia L_world anterior para computar redução
        self._low_novelty_streak: int = 0
        self._ema_stats = {
            "novelty":  {"mean": 0.0, "var": 1.0},
            "progress": {"mean": 0.0, "var": 1.0},
            "surprise": {"mean": 0.0, "var": 1.0},
        }

    def _normalize_signal(self, key: str, value: float) -> float:
        """Normaliza sinal com EMA para reduzir ruído e drift de escala."""
        stats = self._ema_stats[key]
        alpha = self.cfg.reward_ema_alpha

        prev_mean = stats["mean"]
        new_mean = (1.0 - alpha) * prev_mean + alpha * value
        centered = value - new_mean
        new_var = (1.0 - alpha) * stats["var"] + alpha * (centered * centered)

        stats["mean"] = new_mean
        stats["var"] = max(new_var, 1e-6)

        z = centered / (stats["var"] ** 0.5)
        z = max(-self.cfg.reward_norm_clip, min(self.cfg.reward_norm_clip, z))
        # Reescala para [0, 1] de forma estável.
        return 0.5 + 0.5 * (z / self.cfg.reward_norm_clip)

    def compute(
        self,
        Z_t: torch.Tensor,        # estado perceptual atual
        Z_t1: torch.Tensor,       # próximo estado real
        Z_hat_t1: torch.Tensor,   # próximo estado previsto pelo mundo
        V_t: torch.Tensor,        # valor estimado pelo crítico
        world_loss: float,        # L_world atual
    ) -> float:
        """
        Computa r_t e atualiza o erro de mundo anterior.

        Returns:
            r_t: float
        """
        # Novidade: quão diferente é o novo estado
        novelty = torch.norm(Z_t1 - Z_t, p=2).item()

        # Progresso: redução do erro de mundo (max(0, L_world_prev - L_world_t))
        # Um sinal positivo só quando o modelo está realmente aprendendo.
        progress = max(0.0, self._prev_world_loss - world_loss)
        self._prev_world_loss = world_loss

        # Surpresa: erro de previsão do modelo do mundo
        surprise = torch.norm(Z_t1 - Z_hat_t1, p=2).item()

        # Valor do crítico (penalidade por alta estimativa)
        v = V_t.item() if hasattr(V_t, "item") else float(V_t)

        novelty_n = self._normalize_signal("novelty", novelty)
        progress_n = self._normalize_signal("progress", progress)
        surprise_n = self._normalize_signal("surprise", surprise)

        intrinsic_core = (
            self.cfg.alpha_novelty * novelty_n
            + self.cfg.beta_progress * progress_n
            + self.cfg.gamma_surprise * surprise_n
        )

        # Anti-curiosidade-morta: se novidade fica baixa por muito tempo, gera impulso exploratório.
        if novelty < self.cfg.novelty_dead_zone:
            self._low_novelty_streak += 1
        else:
            self._low_novelty_streak = 0

        boredom_scale = min(self._low_novelty_streak, 20) / 20.0
        boredom_drive = self.cfg.boredom_bonus * boredom_scale

        curiosity_drive = max(intrinsic_core + boredom_drive, self.cfg.curiosity_floor)
        r_t = curiosity_drive - self.cfg.lambda_critic * v
        return r_t
