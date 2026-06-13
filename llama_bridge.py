"""
Ponte para o Llama via Ollama API
==================================
Gerencia geração de texto e extração de embeddings usando
o servidor Ollama local.
"""

import hashlib
import struct
import time

import requests
import numpy as np
from typing import Any, Dict, List, Optional

from config import Config

# Dimensão do fallback local (usada quando o modelo não suporta embeddings)
_FALLBACK_DIM = 1024


def _local_embed(text: str, dim: int = _FALLBACK_DIM) -> np.ndarray:
    """
    Embedding determinístico local baseado em hashing.
    Usado como fallback quando o modelo não suporta /api/embed.

    Técnica: Random Projection via hash (SimHash estendido).
    Garante vetores consistentes e normalizados para o mesmo texto.
    """
    vec = np.zeros(dim, dtype=np.float32)
    words = text.lower().split()

    # Bag-of-words com hash
    for i, word in enumerate(words):
        for n in range(1, 4):  # uni, bi, trigrams de chars
            gram = word[:n] if len(word) >= n else word
            seed = gram + str(i % 8)
            h = hashlib.md5(seed.encode()).digest()
            # Mapeia 4 bytes → índice e sinal
            idx = struct.unpack_from("<I", h, 0)[0] % dim
            sign = 1.0 if struct.unpack_from("<I", h, 4)[0] % 2 == 0 else -1.0
            vec[idx] += sign

    # Adiciona hash do texto completo para preservar unicidade global
    full_h = hashlib.sha256(text.encode()).digest()
    for i in range(0, min(32, len(full_h) - 3), 4):
        idx = struct.unpack_from("<I", full_h, i)[0] % dim
        val = struct.unpack_from("<f", full_h, i)[0]
        vec[idx] += np.clip(val, -1.0, 1.0)

    # Normaliza
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


class LlamaBridge:
    """
    Interface com o Llama/Qwen rodando localmente via Ollama.

    Endpoints usados:
      POST /api/embed       → embeddings (API nova)
      POST /api/embeddings  → embeddings (API antiga, fallback)
      POST /api/chat        → geração de texto
      GET  /api/tags        → lista modelos disponíveis

    Se o modelo não suportar embeddings, usa _local_embed() como fallback.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.base_url = cfg.ollama_url.rstrip("/")
        self.model = cfg.llama_model
        self._embed_dim: Optional[int] = None
        self._use_local_embed: bool = False  # ativado se o modelo não suporta embed
        self.embed_timeout = cfg.embed_timeout_seconds
        self.chat_timeout = cfg.chat_timeout_seconds
        self.retries = cfg.ollama_retries
        self.retry_backoff = cfg.ollama_retry_backoff_seconds

    # ──────────────────────────────────────────────────────────────
    # Utilitários
    # ──────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Verifica se o servidor Ollama está rodando."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Lista modelos disponíveis no Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def get_embed_dim(self) -> int:
        """Auto-detecta a dimensão dos embeddings do modelo."""
        if self._embed_dim is not None:
            return self._embed_dim
        emb = self.embed("olá")
        self._embed_dim = len(emb)
        return self._embed_dim

    # ──────────────────────────────────────────────────────────────
    # Embeddings  →  Z_t bruto (antes da projeção do encoder)
    # ──────────────────────────────────────────────────────────────

    def _try_api_embed(self, text: str) -> Optional[np.ndarray]:
        """Tenta /api/embed (Ollama >= 0.3, API nova)."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": text},
                timeout=self.embed_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                # Resposta pode ser {"embeddings": [[...]]} ou {"embedding": [...]}
                if "embeddings" in data:
                    return np.array(data["embeddings"][0], dtype=np.float32)
                if "embedding" in data:
                    return np.array(data["embedding"], dtype=np.float32)
        except Exception:
            pass
        return None

    def _try_api_embeddings(self, text: str) -> Optional[np.ndarray]:
        """Tenta /api/embeddings (Ollama API antiga)."""
        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.embed_timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "embedding" in data:
                    return np.array(data["embedding"], dtype=np.float32)
        except Exception:
            pass
        return None

    def embed(self, text: str) -> np.ndarray:
        """
        Obtém embedding do texto.

        Ordem de tentativas:
          1. /api/embed  (Ollama novo)
          2. /api/embeddings  (Ollama antigo)
          3. _local_embed()  (fallback determinístico local)

        Returns:
            np.ndarray de shape [embed_dim], dtype float32
        """
        if self._use_local_embed:
            return _local_embed(text, _FALLBACK_DIM)

        # Tenta API nova
        result = self._try_api_embed(text)
        if result is not None:
            return result

        # Tenta API antiga
        result = self._try_api_embeddings(text)
        if result is not None:
            return result

        # Fallback local — modelo não suporta embeddings
        print(
            f"  [embed] Modelo '{self.model}' não suporta embeddings via Ollama. "
            "Usando embedding local determinístico."
        )
        self._use_local_embed = True
        return _local_embed(text, _FALLBACK_DIM)

    # ──────────────────────────────────────────────────────────────
    # Geração de texto  →  a_t^ext (em linguagem natural)
    # ──────────────────────────────────────────────────────────────

    def _action_embedding_to_prefix(self, action_embedding: Optional[np.ndarray]) -> str:
        """
        Converte vetor de ação (a_ext) em um prefixo de estilo/personalidade
        que condiciona a geração do LLM.

        A ação é mapeada para dimensões semânticas que influenciam o tom:
          - Dimensão 0-85: verbose vs. conciso
          - Dimensão 85-170: curioso vs. pragmático
          - Dimensão 170-256: confiante vs. hesitante

        Args:
            action_embedding: numpy array [256], não normalizado

        Returns:
            str com prefixo de instrução que guia o tom/estilo
        """
        if action_embedding is None:
            return ""

        a = action_embedding.astype(np.float32)
        
        # Quantiza em 3 dimensões de estilo
        verbose_score = float(np.clip(np.mean(a[0:85]) / (np.std(a[0:85]) + 1e-6), -1, 1))
        curiosity_score = float(np.clip(np.mean(a[85:170]) / (np.std(a[85:170]) + 1e-6), -1, 1))
        confidence_score = float(np.clip(np.mean(a[170:256]) / (np.std(a[170:256]) + 1e-6), -1, 1))

        # Constrói prefixo descritivo baseado na ação
        prefix_parts = []

        # Tone instrução baseada em confidence_score
        if confidence_score > 0.3:
            prefix_parts.append("Responda com confiança e clareza.")
        elif confidence_score < -0.3:
            prefix_parts.append("Responda de forma reflexiva e humilde, considerando incertezas.")
        else:
            prefix_parts.append("Responda de forma equilibrada.")

        # Estilo baseado em verbosity
        if verbose_score > 0.3:
            prefix_parts.append("Use explicações detalhadas e exemplos concretos.")
        elif verbose_score < -0.3:
            prefix_parts.append("Seja conciso e direto ao ponto.")
        else:
            prefix_parts.append("Mantenha um nível de detalhe moderado.")

        # Abordagem baseada em curiosidade
        if curiosity_score > 0.3:
            prefix_parts.append("Explore aspectos interessantes e faça perguntas de acompanhamento.")
        elif curiosity_score < -0.3:
            prefix_parts.append("Foque em respostas pragmáticas e diretas.")
        else:
            prefix_parts.append("Balanceie pragmatismo e exploração.")

        return " ".join(prefix_parts)

    def generate(
        self,
        user_message: str,
        system_prompt: str = "",
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.85,
        max_tokens: int = 1024,
        action_embedding: Optional[np.ndarray] = None,
    ) -> str:
        """
        Gera resposta usando o Llama via Ollama chat API, com steering opcional via a_ext.

        Args:
            user_message: mensagem do usuário
            system_prompt: contexto de sistema (inclui memórias relevantes)
            history: histórico de mensagens [{"role":…, "content":…}]
            temperature: temperatura de amostragem τ
            max_tokens: limite de tokens gerados
            action_embedding: vetor a_ext [256] para condicionamento do estilo (opcional)

        Returns:
            Texto da resposta gerada
        """
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        # Injeta prefixo de steering no user_message se action_embedding fornecido
        steering_prefix = self._action_embedding_to_prefix(action_embedding)
        if steering_prefix:
            user_message = f"[INSTRUÇÃO DE ESTILO: {steering_prefix}]\n\n{user_message}"

        messages.append({"role": "user", "content": user_message})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.chat_timeout,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except requests.exceptions.Timeout as e:
                last_error = e
                if attempt < self.retries:
                    wait_s = self.retry_backoff * (attempt + 1)
                    print(
                        f"  [chat] Timeout na tentativa {attempt + 1}/"
                        f"{self.retries + 1}. Aguardando {wait_s:.1f}s e tentando novamente..."
                    )
                    time.sleep(wait_s)
                    continue
                break
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.retries:
                    wait_s = self.retry_backoff * (attempt + 1)
                    print(
                        f"  [chat] Erro de rede na tentativa {attempt + 1}/"
                        f"{self.retries + 1}: {e}. Tentando novamente em {wait_s:.1f}s..."
                    )
                    time.sleep(wait_s)
                    continue
                break

        raise RuntimeError(
            "Falha ao gerar resposta no Ollama após múltiplas tentativas. "
            f"Modelo: '{self.model}', timeout: {self.chat_timeout}s, retries: {self.retries}. "
            f"Erro final: {last_error}"
        )
