"""Analise e rascunho para posts do Reddit usando a Deorita."""

from __future__ import annotations

import json
import re
from typing import Dict


class DeoritaRedditBrain:
    def __init__(self, agent):
        self.agent = agent

    def analyze_post(self, post_text: str) -> Dict[str, object]:
        prompt = (
            "Analise este post do Reddit e responda APENAS em JSON com as chaves: "
            "interest_level, should_like, should_comment, reason.\n\n"
            f"Post: {post_text[:1200]}\n"
            f"Personalidade atual: {self.agent._personality_to_text()}"
        )
        raw = self.agent.llama.generate(
            user_message=prompt,
            system_prompt="Seja objetiva e responda somente JSON valido.",
            history=[],
            temperature=0.35,
            max_tokens=140,
        )
        return self._parse_jsonish(raw)

    def draft_comment(self, post_text: str) -> str:
        prompt = (
            "Escreva um comentario curto para este post do Reddit. "
            "Precisa soar humano, coerente e compativel com sua personalidade atual. "
            "Maximo 500 caracteres.\n\n"
            f"Post: {post_text[:1200]}"
        )
        return self.agent.llama.generate(
            user_message=prompt,
            system_prompt="Escreva apenas o comentario final, sem aspas e sem explicacoes.",
            history=[],
            temperature=0.65,
            max_tokens=180,
        ).strip()

    def _parse_jsonish(self, raw: str) -> Dict[str, object]:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                return {
                    "interest_level": float(max(0.0, min(1.0, float(data.get("interest_level", 0.5))))),
                    "should_like": bool(data.get("should_like", False)),
                    "should_comment": bool(data.get("should_comment", False)),
                    "reason": str(data.get("reason", "Analise indisponivel."))[:220],
                }
            except Exception:
                pass
        return {
            "interest_level": 0.5,
            "should_like": False,
            "should_comment": False,
            "reason": raw.strip()[:220] or "Analise indisponivel.",
        }