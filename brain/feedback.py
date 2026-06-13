"""
Feedback Cíclico — EcoMental avalia a si mesma
================================================
A cada N passos, a IA gera uma autoavaliação estruturada da sua última
interação e usa o resultado para:

  1. Ajustar o multiplicador de recompensa intrínseca do próximo passo
     → qualidade alta  →  r_t * boost    (aprendizado mais rápido)
     → qualidade baixa →  r_t * penalidade (sinal de que algo está errado)

  2. Guardar o aprendizado em memória como fato IMPORTANTE

  3. Fornecer contexto comportamental ao system prompt:
     "Minha última autoavaliação: ..."

Estrutura do feedback:
    QUALIDADE: 0-10   (quão boa foi a resposta globalmente)
    CURIOSIDADE: 0-10 (quão exploratória/curiosa)
    COERENCIA: 0-10   (consistência com o contexto e identidade)
    APRENDI: <frase>   (o que foi absorvido nessa troca)
    MELHORAR: <frase>  (o que fazer diferente próxima vez)
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_bridge import LlamaBridge


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura de uma entrada de feedback
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedbackEntry:
    step:       int
    quality:    float   # 0–1  (normalizado de 0–10)
    curiosity:  float   # 0–1
    coherence:  float   # 0–1
    learned:    str
    improve:    str
    raw:        str = field(default="", repr=False)

    @property
    def composite(self) -> float:
        """Pontuação composta: média ponderada dos três eixos."""
        return 0.5 * self.quality + 0.3 * self.curiosity + 0.2 * self.coherence


# ─────────────────────────────────────────────────────────────────────────────
# Motor de Feedback Cíclico
# ─────────────────────────────────────────────────────────────────────────────

class CyclicFeedback:
    """
    Mantém histórico de autoavaliações e calcula modificadores de recompensa.

    Parâmetros
    ----------
    llama        : ponte para o modelo de linguagem
    eval_every   : número de passos entre avaliações
    max_history  : quantas entradas de feedback guardar em memória
    boost_max    : multiplicador máximo quando qualidade cresce
    penalty_max  : multiplicador mínimo quando qualidade cai
    """

    def __init__(
        self,
        llama: "LlamaBridge",
        eval_every: int = 3,
        max_history: int = 20,
        boost_max: float = 1.35,
        penalty_max: float = 0.72,
    ):
        self.llama        = llama
        self.eval_every   = eval_every
        self.max_history  = max_history
        self.boost_max    = boost_max
        self.penalty_max  = penalty_max

        self.history: Deque[FeedbackEntry] = deque(maxlen=max_history)
        self._reward_modifier: float = 1.0   # aplicado ao próximo passo
        self._pending_memory:  Optional[str] = None  # texto a armazenar

    # ──────────────────────────────────────────────────────────────
    # Avaliação
    # ──────────────────────────────────────────────────────────────

    def should_evaluate(self, step: int) -> bool:
        return step % self.eval_every == 0

    def evaluate(
        self,
        user_input: str,
        response: str,
        step: int,
        r_t: float,
    ) -> Optional[FeedbackEntry]:
        """
        Pede ao Llama que avalie a própria última troca.
        Retorna None se a chamada falhar.
        """
        prompt = (
            f"Você acabou de interagir.\n\n"
            f"Pergunta recebida:\n{user_input[:300]}\n\n"
            f"Sua resposta:\n{response[:500]}\n\n"
            f"Avalie honestamente esta troca. Responda APENAS neste formato, "
            f"sem texto antes ou depois:\n"
            f"QUALIDADE: <0-10>\n"
            f"CURIOSIDADE: <0-10>\n"
            f"COERENCIA: <0-10>\n"
            f"APRENDI: <uma frase curta>\n"
            f"MELHORAR: <uma frase curta>"
        )

        system = (
            "Você é EcoMental em modo de introspecção. "
            "Seja brutalmente honesta consigo mesma. "
            "Responda APENAS com as 5 linhas pedidas, sem introduções."
        )

        try:
            raw = self.llama.generate(
                user_message  = prompt,
                system_prompt = system,
                history       = [],
                temperature   = 0.4,
                max_tokens    = 90,
            )
        except Exception:
            return None

        try:
            entry = self._parse(raw, step)
        except Exception:
            return None

        if entry is None:
            return None

        self.history.append(entry)
        self._update_modifier()
        self._pending_memory = (
            f"[Autoavaliação passo {step}] "
            f"qualidade={entry.quality:.2f} coerência={entry.coherence:.2f} | "
            f"aprendi: {entry.learned} | melhorar: {entry.improve}"
        )
        return entry

    # ──────────────────────────────────────────────────────────────
    # Parse da resposta estruturada
    # ──────────────────────────────────────────────────────────────

    def _parse(self, raw: str, step: int) -> Optional[FeedbackEntry]:
        """Extrai campos do formato esperado. Tolerante a erros de formato."""
        def extract_num(label: str) -> Optional[float]:
            # Envolve o label em (?:...) para evitar que alternações quebrem o grupo capturador
            m = re.search(rf"(?:{label})\s*:\s*([0-9]+(?:\.[0-9]+)?)", raw, re.IGNORECASE)
            if not m or m.group(1) is None:
                return None
            try:
                val = float(m.group(1))
            except (ValueError, TypeError):
                return None
            return max(0.0, min(val / 10.0, 1.0))  # normaliza 0-10 → 0-1

        def extract_text(label: str) -> str:
            m = re.search(rf"{label}\s*:\s*(.+)", raw, re.IGNORECASE)
            if not m:
                return ""
            return m.group(1).strip()[:200]

        quality   = extract_num("QUALIDADE")
        curiosity = extract_num("CURIOSIDADE")
        coherence = extract_num("COERENCIA|COERÊNCIA")
        learned   = extract_text("APRENDI")
        improve   = extract_text("MELHORAR")

        # Se não conseguiu extrair ao menos qualidade, falha
        if quality is None:
            return None

        return FeedbackEntry(
            step      = step,
            quality   = quality,
            curiosity = curiosity if curiosity is not None else 0.5,
            coherence = coherence if coherence is not None else 0.5,
            learned   = learned   or "—",
            improve   = improve   or "—",
            raw       = raw,
        )

    # ──────────────────────────────────────────────────────────────
    # Modificador de recompensa
    # ──────────────────────────────────────────────────────────────

    def _update_modifier(self):
        """
        Calcula modificador para o próximo r_t baseado na tendência recente.

        Se a qualidade está subindo  → boost  (até boost_max)
        Se a qualidade está caindo   → penalidade (até penalty_max)
        Se estável ou histórico curto → neutro (1.0)
        """
        if len(self.history) < 2:
            self._reward_modifier = 1.0
            return

        recent = list(self.history)[-5:]          # últimas 5 entradas
        scores = [e.composite for e in recent]

        if len(scores) >= 2:
            # Tendência linear simples
            trend = scores[-1] - scores[0]        # delta total nas últimas 5
        else:
            trend = 0.0

        if trend > 0.05:
            # Melhorando — reforça aprendizado
            ratio = min(trend / 0.3, 1.0)
            self._reward_modifier = 1.0 + ratio * (self.boost_max - 1.0)
        elif trend < -0.05:
            # Piorando — sinal de alerta
            ratio = min(abs(trend) / 0.3, 1.0)
            self._reward_modifier = 1.0 - ratio * (1.0 - self.penalty_max)
        else:
            self._reward_modifier = 1.0

    def get_reward_modifier(self) -> float:
        """Retorna o modificador de recompensa calculado na última avaliação."""
        return self._reward_modifier

    def consume_pending_memory(self) -> Optional[str]:
        """
        Retorna o texto de autoavaliação pendente para armazenar em memória
        e limpa o buffer. Retorna None se não houver nada pendente.
        """
        text = self._pending_memory
        self._pending_memory = None
        return text

    # ──────────────────────────────────────────────────────────────
    # Contexto para o system prompt
    # ──────────────────────────────────────────────────────────────

    def get_behavior_summary(self) -> str:
        """
        Resumo comportamental das últimas avaliações para o system prompt.
        Retorna string vazia se não houver histórico.
        """
        if not self.history:
            return ""

        last = self.history[-1]
        trend = self.get_trend()
        return (
            f"[Autoavaliação recente — passo {last.step}] "
            f"qualidade={last.quality:.1f} | curiosidade={last.curiosity:.1f} | "
            f"coerência={last.coherence:.1f} | tendência: {trend} | "
            f"aprendi: {last.learned} | melhorar: {last.improve}"
        )

    def get_trend(self) -> str:
        """Tendência das últimas avaliações: 'melhorando' | 'estável' | 'piorando'."""
        if len(self.history) < 2:
            return "sem dados"
        recent = [e.composite for e in list(self.history)[-5:]]
        delta = recent[-1] - recent[0]
        if delta > 0.05:
            return "melhorando"
        elif delta < -0.05:
            return "piorando"
        return "estável"

    # ──────────────────────────────────────────────────────────────
    # Estatísticas
    # ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Estatísticas do histórico de feedback."""
        if not self.history:
            return {"total_avaliacoes": 0}
        scores = [e.composite for e in self.history]
        return {
            "total_avaliacoes": len(self.history),
            "qualidade_media":  round(sum(e.quality   for e in self.history) / len(self.history), 3),
            "curiosidade_media":round(sum(e.curiosity for e in self.history) / len(self.history), 3),
            "coerencia_media":  round(sum(e.coherence for e in self.history) / len(self.history), 3),
            "composta_media":   round(sum(scores) / len(scores), 3),
            "tendencia":        self.get_trend(),
            "modificador_atual":round(self._reward_modifier, 3),
            "ultimo_aprendizado": self.history[-1].learned if self.history else "—",
            "ultimo_melhorar":    self.history[-1].improve if self.history else "—",
        }

    def get_full_log(self, last_n: int = 10) -> List[FeedbackEntry]:
        """Retorna as últimas N entradas de feedback."""
        return list(self.history)[-last_n:]

    # ──────────────────────────────────────────────────────────────
    # Persistência
    # ──────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serializa estado do feedback para checkpoint do agente."""
        return {
            "eval_every": self.eval_every,
            "max_history": self.max_history,
            "boost_max": self.boost_max,
            "penalty_max": self.penalty_max,
            "reward_modifier": self._reward_modifier,
            "pending_memory": self._pending_memory,
            "history": [
                {
                    "step": entry.step,
                    "quality": entry.quality,
                    "curiosity": entry.curiosity,
                    "coherence": entry.coherence,
                    "learned": entry.learned,
                    "improve": entry.improve,
                    "raw": entry.raw,
                }
                for entry in self.history
            ],
        }

    def load_from_dict(self, state: Optional[dict]):
        """Restaura estado serializado do feedback com fallback seguro."""
        if not state or not isinstance(state, dict):
            return

        self.eval_every = int(state.get("eval_every", self.eval_every))
        self.max_history = int(state.get("max_history", self.max_history))
        self.boost_max = float(state.get("boost_max", self.boost_max))
        self.penalty_max = float(state.get("penalty_max", self.penalty_max))
        self._reward_modifier = float(state.get("reward_modifier", self._reward_modifier))
        self._pending_memory = state.get("pending_memory", self._pending_memory)

        raw_history = state.get("history", [])
        restored: Deque[FeedbackEntry] = deque(maxlen=max(1, self.max_history))
        for item in raw_history:
            if not isinstance(item, dict):
                continue
            try:
                restored.append(
                    FeedbackEntry(
                        step=int(item.get("step", 0)),
                        quality=float(item.get("quality", 0.5)),
                        curiosity=float(item.get("curiosity", 0.5)),
                        coherence=float(item.get("coherence", 0.5)),
                        learned=str(item.get("learned", "—"))[:200],
                        improve=str(item.get("improve", "—"))[:200],
                        raw=str(item.get("raw", ""))[:1200],
                    )
                )
            except Exception:
                continue

        self.history = restored
        self._update_modifier()
