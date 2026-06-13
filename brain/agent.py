"""
Agente Central — Cérebro do EcoMental
=======================================
Orquestra todos os componentes do sistema numa única classe.

Fluxo de informação por passo:

  x_t → Encoder → Z_t
                    ↓
              Recuperação Memória → R_t
                    ↓
         [Loop Interno K vezes]
         S_{t,k-1} + Z_t + R_t → π_int → a_int → f_dyn → S_{t,k}
                    ↓
                S_t* = S_{t,K}
                    ↓
         π_ext(S_t*, Z_t, R_t) + ε → a_t^ext
                    ↓
         Llama.generate(prompt_aumentado) → resposta
                    ↓
         f_enc(resposta) → Z_{t+1}
                    ↓
         f_dyn(S_t*, Z_t, a_ext, R_t) → S_{t+1}
                    ↓
         Cálculo de r_t, g_t, L_total
                    ↓
         Atualização via ∇L   +   Memória
"""

import os
import re
from datetime import datetime
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from llama_bridge import LlamaBridge
from brain.encoder import WorldEncoder
from brain.memory import EpisodicMemory
from brain.dynamics import StateDynamics, SelfPredictor
from brain.world_model import WorldModel
from brain.critic import Critic
from brain.policy import InternalPolicy, ExternalPolicy
from brain.personality import Personality
from brain.rewards import IntrinsicRewards
from brain.gate import LearningGate
from brain.feedback import CyclicFeedback


class CentralAgent:
    """Agente Central — integra Llama com o framework matemático completo."""

    def __init__(self, cfg: Config, llama: LlamaBridge):
        self.cfg    = cfg
        self.llama  = llama
        self.device = cfg.device

        # ── Auto-detectar dimensão dos embeddings do Llama ────────
        print(f"  Detectando dimensão de embedding de '{cfg.llama_model}'...")
        try:
            embed_dim = llama.get_embed_dim()
            print(f"  Dimensão detectada: {embed_dim}")
        except Exception as e:
            embed_dim = cfg.llama_embed_dim
            print(f"  Usando padrão: {embed_dim}  ({e})")

        # ── Redes Neurais ─────────────────────────────────────────
        self.encoder       = WorldEncoder(cfg, llama_embed_dim=embed_dim).to(self.device)
        self.dynamics      = StateDynamics(cfg).to(self.device)
        self.self_pred     = SelfPredictor(cfg).to(self.device)
        self.world_model   = WorldModel(cfg).to(self.device)
        self.critic        = Critic(cfg).to(self.device)
        self.pi_int        = InternalPolicy(cfg).to(self.device)
        self.pi_ext        = ExternalPolicy(cfg).to(self.device)
        self.personality   = Personality(cfg).to(self.device)
        self.gate          = LearningGate(cfg).to(self.device)

        # ── Otimizador unificado ──────────────────────────────────
        all_params = (
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.self_pred.parameters())
            + list(self.world_model.parameters())
            + list(self.critic.parameters())
            + list(self.pi_int.parameters())
            + list(self.pi_ext.parameters())
            + list(self.gate.parameters())
        )
        self.optimizer = optim.AdamW(all_params, lr=cfg.lr, weight_decay=1e-5)
        self.personality_optimizer = optim.Adam([self.personality.P], lr=cfg.personality_lr)

        # ── Componentes não-neurais ───────────────────────────────
        self.memory     = EpisodicMemory(cfg)
        self.reward_fn  = IntrinsicRewards(cfg)

        # ── Estado persistente do agente ──────────────────────────
        self.S_state = torch.zeros(cfg.d_s, device=self.device)

        # ── Histórico de conversa para o Llama ───────────────────
        self.chat_history: List[Dict] = []
        self.max_history  = 10

        # Replay de transições para reduzir esquecimento catastrófico online.
        self.replay_buffer = deque(maxlen=cfg.replay_capacity)

        # ── Contadores ────────────────────────────────────────────
        self.step_count   = 0
        self.total_reward = 0.0
        self.low_gate_streak = 0

        # ── Feedback Cíclico (autoavaliação) ─────────────────────
        self.feedback = CyclicFeedback(
            llama       = llama,
            eval_every  = 3,       # avalia a cada 3 passos
            max_history = 20,
            boost_max   = 1.35,
            penalty_max = 0.72,
        )

        # ── Busca Web ─────────────────────────────────────────────
        self.autonomous_search = cfg.autonomous_search_enabled and cfg.web_search_enabled
        self.web_search_count = 0
        self.last_search_query = ""
        self.last_search_summary = ""
        self.self_talk_events = 0
        self.last_self_talk_reason = ""
        self.self_name = "deorita"
        self.self_marker = f"{self.self_name}:"
        self.self_dialogue_tag = "[AUTO_DIALOGO_INTERNO]"

    def _has_self_marker(self, text: str) -> bool:
        """Identifica marcador explicito de fala da propria IA."""
        raw = (text or "").strip().lower()
        return raw.startswith(self.self_marker) or raw.startswith(f"[{self.self_name}]")

    def _detect_self_talk(self, user_input: str) -> Tuple[bool, str, float]:
        """Detecta auto-conversa por marcador e padroes semanticos de dialogo interno."""
        raw = (user_input or "").strip()
        raw_lower = raw.lower()

        if self.self_dialogue_tag.lower() in raw_lower:
            return True, "tag de dialogo interno", 1.0

        if self._has_self_marker(raw):
            return True, "marcador deorita identificado", 1.0

        semantic_patterns = [
            r"\bresposta anterior\b",
            r"\bminha pergunta\b",
            r"\bvoce mesma\b",
            r"\bdi[aá]logo interno\b",
            r"\bauto[- ]?conversa\b",
            r"\bautodi[aá]logo\b",
        ]
        hits = sum(1 for p in semantic_patterns if re.search(p, raw_lower))

        if hits >= 1:
            confidence = min(0.55 + 0.12 * hits, 0.95)
            return True, f"padrao semantico de auto-conversa ({hits})", confidence

        return False, "", 0.0

    # ──────────────────────────────────────────────────────────────
    # Percepção
    # ──────────────────────────────────────────────────────────────

    def _perceive(self, text: str) -> torch.Tensor:
        """Z_t = f_enc( Llama_embed(text) )"""
        raw = self.llama.embed(text)
        return self.encoder.encode_numpy(raw, device=self.device)

    # ──────────────────────────────────────────────────────────────
    # Voz Interna — K passos de raciocínio
    # ──────────────────────────────────────────────────────────────

    def _select_thinking_depth(self, user_input: str, relevance: float) -> int:
        """Seleciona K dinamicamente com base em complexidade + incerteza contextual."""
        k_min = self.cfg.adaptive_k_min
        k_max = self.cfg.adaptive_k_max
        if k_min >= k_max:
            return max(1, self.cfg.K)

        # Entradas longas tendem a exigir mais passos.
        input_complexity = min(len(user_input) / 240.0, 1.0)

        # Relevância baixa indica contexto menos estável; precisa pensar mais.
        relevance_norm = max(0.0, min(1.0, (relevance + 1.0) / 2.0))
        uncertainty_proxy = 1.0 - relevance_norm

        score = (
            self.cfg.adaptive_k_relevance_weight * uncertainty_proxy
            + self.cfg.adaptive_k_input_weight * input_complexity
        )
        k = int(round(k_min + score * (k_max - k_min)))
        return max(k_min, min(k, k_max))

    def _inner_voice(self, Z_t: torch.Tensor, R_t: torch.Tensor, thinking_steps: int) -> torch.Tensor:
        """
        Para k = 1..K:
            a_int = π_int(S_{t,k-1}, R_t)
            S_{t,k} = f_dyn(S_{t,k-1}, Z_t, a_int, R_t)
        Retorna S_t* = S_{t,K}
        """
        S = self.S_state
        for _ in range(thinking_steps):
            a_int = self.pi_int(S, R_t)
            S = self.dynamics(S, Z_t, a_int, R_t)
        return S

    def _select_context_memory_count(self, user_input: str, relevance: float, candidate_texts: List[str]) -> int:
        """Define quantas memórias entram no prompt para equilibrar qualidade e latência."""
        max_k = max(1, int(self.cfg.max_context_memories))
        min_k = max(1, min(int(self.cfg.min_context_memories), max_k))

        if not candidate_texts:
            return 0

        if not self.cfg.adaptive_context_memories:
            return min(max_k, len(candidate_texts))

        # Mais relevância e maior complexidade de entrada tendem a exigir mais contexto.
        input_complexity = min(len(user_input) / 260.0, 1.0)
        relevance_norm = max(0.0, min(1.0, (relevance + 1.0) / 2.0))
        ratio = 0.65 * relevance_norm + 0.35 * input_complexity
        desired_k = int(round(min_k + ratio * (max_k - min_k)))
        desired_k = max(min_k, min(desired_k, max_k, len(candidate_texts)))

        # Limite de tamanho do contexto para não degradar tempo por excesso de tokens.
        target_chars = max(200, int(self.cfg.context_target_chars))
        char_budget = 0
        by_chars = 0
        for text in candidate_texts[:max_k]:
            char_budget += min(len(text), 200)
            if char_budget > target_chars:
                break
            by_chars += 1

        if by_chars == 0:
            by_chars = 1

        return max(1, min(desired_k, by_chars, len(candidate_texts)))

    # ──────────────────────────────────────────────────────────────
    # Autonomia de Memória — a IA revisa e decide o que guardar
    # ──────────────────────────────────────────────────────────────

    def _autonomous_memory_review(self):
        """
        A IA revisa suas memórias recentes e decide, ela mesma, o que guardar
        fortemente e o que pode deixar ir.

        Mecanismo:
          1. Recupera candidatos (não-pinned) ordenados por prioridade
          2. Monta prompt conciso para o Llama
          3. Parseia resposta: GUARDAR: 1,3  | SOLTAR: 2,5
          4. Aplica protect_memory / release_memory nos índices escolhidos
        """
        candidates = self.memory.get_review_candidates(k=8)
        if len(candidates) < 2:
            return  # Pouca memória ainda — sem necessidade de revisão

        # Formata lista de memórias para a IA avaliar
        lines = []
        for pos, (idx, text, prio, status) in enumerate(candidates, start=1):
            marker = " [guardando]" if status == "guardar" else (" [soltando]" if status == "soltar" else "")
            lines.append(f"{pos}. {text[:130]}{marker}")
        mem_list = "\n".join(lines)

        review_prompt = (
            f"Estas são suas memórias recentes (ordenadas por relevância):\n\n"
            f"{mem_list}\n\n"
            f"Decida agora:\n"
            f"- Quais você quer GUARDAR com força (vai decair muito devagar)?\n"
            f"- Quais você quer SOLTAR (pode deixar ir, não precisa mais)?\n"
            f"- O restante fica com o decay normal.\n\n"
            f"Responda APENAS neste formato exato, sem explicações:\n"
            f"GUARDAR: <números separados por vírgula, ou NENHUM>\n"
            f"SOLTAR: <números separados por vírgula, ou NENHUM>"
        )

        try:
            raw = self.llama.generate(
                user_message  = review_prompt,
                system_prompt = (
                    "Você é EcoMental. Responda APENAS com as duas linhas pedidas. "
                    "Sem comentários extras."
                ),
                history       = [],
                temperature   = 0.3,
                max_tokens    = 80,
            )
        except Exception:
            return  # Falha silenciosa — não interrompe o fluxo principal

        # Parseia GUARDAR e SOLTAR
        guardar_idxs: List[int] = []
        soltar_idxs:  List[int] = []
        for line in raw.splitlines():
            line_up = line.strip().upper()
            if line_up.startswith("GUARDAR:"):
                nums_str = line.split(":", 1)[1].strip()
                if nums_str.upper() != "NENHUM":
                    for tok in nums_str.replace(";", ",").split(","):
                        tok = tok.strip()
                        if tok.isdigit():
                            pos = int(tok) - 1  # 1-based → 0-based
                            if 0 <= pos < len(candidates):
                                guardar_idxs.append(candidates[pos][0])
            elif line_up.startswith("SOLTAR:"):
                nums_str = line.split(":", 1)[1].strip()
                if nums_str.upper() != "NENHUM":
                    for tok in nums_str.replace(";", ",").split(","):
                        tok = tok.strip()
                        if tok.isdigit():
                            pos = int(tok) - 1
                            if 0 <= pos < len(candidates):
                                soltar_idxs.append(candidates[pos][0])

        for idx in guardar_idxs:
            self.memory.protect_memory(idx)
        for idx in soltar_idxs:
            self.memory.release_memory(idx)

    def _get_searcher(self):
        from web_search import get_searcher

        return get_searcher(
            max_results=self.cfg.web_search_max_results,
            timeout=self.cfg.web_search_timeout_seconds,
            page_max_chars=self.cfg.web_page_max_chars,
        )

    def search_web(self, query: str, read_top_n: Optional[int] = None, store: bool = True) -> str:
        """Executa busca web e opcionalmente grava o resultado na memoria."""
        if not self.cfg.web_search_enabled:
            return "Busca web desativada na configuracao."

        query = query.strip()
        if not query:
            return "Consulta vazia."

        searcher = self._get_searcher()
        result = searcher.search_and_read(
            query,
            read_top_n=read_top_n or self.cfg.web_read_top_n,
            max_results=self.cfg.web_search_max_results,
        )

        self.web_search_count += 1
        self.last_search_query = query
        self.last_search_summary = result[:600]

        if store:
            Z_q = self._perceive(query)
            self.memory.add(
                Z=Z_q,
                text=f"[PESQUISA WEB] Query: {query}\n{result[:1200]}",
                r_t=0.45,
                relevance=0.70,
                impact=0.75,
            )
        return result

    def chat_with_external_ai(self, url: str, user_message: str, store: bool = True) -> str:
        """Conversa com IA externa hospedada em página web e opcionalmente grava na memória."""
        try:
            from web_automation import interact_with_web_ai
        except ImportError:
            return "Automação web não configurada. Execute: pip install playwright && playwright install"

        url = url.strip()
        user_message = user_message.strip()

        if not url or not user_message:
            return "URL e mensagem são obrigatórias."

        print(f"  [web_automation] Iniciando interação com {url[:50]}...")

        response = interact_with_web_ai(
            url=url,
            message=user_message,
            headless=True,
            timeout=30,
        )

        if response is None:
            return "Falha ao obter resposta da IA externa. Verifique URL e tente novamente."

        if store:
            Z_q = self._perceive(f"Conversa com IA externa: {user_message}")
            self.memory.add(
                Z=Z_q,
                text=f"[IA EXTERNA] URL: {url[:100]}\nMinha pergunta: {user_message[:200]}\nResposta: {response[:500]}",
                r_t=0.55,
                relevance=0.75,
                impact=0.70,
            )

        return response

    def chat_with_external_ai_clickbot(self, user_message: str, store: bool = True) -> str:
        """Conversa com IA externa por automacao mecanica baseada em templates de imagem."""
        try:
            from image_click_bot import mechanical_web_chat
        except ImportError:
            return "Bot mecanico não configurado. Instale: pyautogui, pyperclip e pillow."

        user_message = (user_message or "").strip()
        if not user_message:
            return "Mensagem vazia."

        print("  [click_bot] Iniciando fluxo por imagem...")
        response = mechanical_web_chat(user_message)

        if not response:
            return "Falha ao obter resposta da IA externa via bot mecanico."

        if store:
            Z_q = self._perceive(f"Conversa externa (clickbot): {user_message}")
            self.memory.add(
                Z=Z_q,
                text=(
                    f"[IA EXTERNA CLICKBOT]\n"
                    f"Pergunta: {user_message[:220]}\n"
                    f"Resposta: {response[:700]}"
                ),
                r_t=0.52,
                relevance=0.74,
                impact=0.72,
            )

        return response

    def _decide_autonomous_search_query(self) -> Optional[str]:
        """Deixa a IA decidir se quer pesquisar algo agora."""
        if not self.cfg.web_search_enabled:
            return None

        recent = self.memory.get_pinned_texts(max_items=2) + self.memory.texts[-3:]
        recent_context = "\n".join(t[:180] for t in recent[-5:]) if recent else "Sem contexto recente relevante."
        prompt = (
            "Voce pode pesquisar na web para aprender algo novo. "
            "Analise seu contexto recente e decida se existe uma lacuna real de conhecimento.\n"
            "Se SIM, responda APENAS com uma consulta objetiva de busca.\n"
            "Se NAO, responda APENAS: NADA\n\n"
            f"Contexto recente:\n{recent_context}"
        )
        try:
            decision = self.llama.generate(
                user_message=prompt,
                system_prompt="Voce decide pesquisas de forma pragmatica e curiosa.",
                history=[],
                temperature=0.35,
                max_tokens=60,
            ).strip()
        except Exception:
            return None

        if not decision or decision.upper().startswith("NADA"):
            return None

        decision = decision.replace("\n", " ").strip(" \t-:;,.\"'")
        if len(decision) < 4:
            return None
        return decision[:120]

    def _autonomous_search_tick(self) -> Optional[Dict[str, str]]:
        """Executa pesquisa autonoma periodica e salva o aprendizado na memoria."""
        if not self.autonomous_search or not self.cfg.web_search_enabled:
            return None
        if self.cfg.autonomous_search_every <= 0:
            return None
        if self.step_count % self.cfg.autonomous_search_every != 0:
            return None

        query = self._decide_autonomous_search_query()
        if not query:
            return None

        result = self.search_web(query, store=False)
        reflection_prompt = (
            f"Consulta: {query}\n"
            f"Resultado bruto:\n{result[:1200]}\n\n"
            "Resuma em 2 frases o que aprendi e por que isso importa para mim agora."
        )
        try:
            reflection = self.llama.generate(
                user_message=reflection_prompt,
                system_prompt="Resuma de forma clara e autonoma.",
                history=[],
                temperature=0.4,
                max_tokens=120,
            ).strip()
        except Exception:
            reflection = "Aprendi algo util, mas ainda preciso integrar melhor esse conhecimento."

        memory_text = (
            f"[PESQUISA AUTONOMA] Query: {query}\n"
            f"Aprendizado: {reflection[:300]}\n"
            f"Fonte resumida: {result[:700]}"
        )
        Z_q = self._perceive(query)
        self.memory.add(
            Z=Z_q,
            text=memory_text,
            r_t=0.65,
            relevance=0.75,
            impact=0.80,
        )

        self.web_search_count += 1
        self.last_search_query = query
        self.last_search_summary = reflection[:400]
        return {"query": query, "reflection": reflection[:300], "result": result[:700]}

    def run_autonomous_research(self, num_queries: int = 3) -> List[Dict[str, str]]:
        """Executa uma sessao de pesquisa autonoma deliberada."""
        records: List[Dict[str, str]] = []
        num_queries = max(1, min(int(num_queries), 8))

        for _ in range(num_queries):
            topic_prompt = (
                "Com base na sua memoria recente e curiosidade atual, escolha um topico que voce quer pesquisar agora. "
                "Responda apenas com o topico."
            )
            topic = self.llama.generate(
                user_message=topic_prompt,
                system_prompt="Voce e uma IA curiosa e focada.",
                history=[],
                temperature=0.6,
                max_tokens=40,
            ).strip()
            if not topic:
                topic = "tema ainda indefinido"

            query_prompt = (
                f"Topico escolhido: {topic}\n"
                "Formule uma consulta especifica de busca para aprender algo util."
            )
            query = self.llama.generate(
                user_message=query_prompt,
                system_prompt="Formule consultas objetivas.",
                history=[],
                temperature=0.55,
                max_tokens=50,
            ).strip()
            if not query:
                query = topic

            result = self.search_web(query, store=False)
            reflection_prompt = (
                f"Topico: {topic}\nConsulta: {query}\nResultado:\n{result[:1200]}\n\n"
                "O que voce aprendeu e como isso muda seu entendimento atual?"
            )
            reflection = self.llama.generate(
                user_message=reflection_prompt,
                system_prompt="Resuma o aprendizado em um paragrafo curto.",
                history=[],
                temperature=0.5,
                max_tokens=160,
            ).strip()

            Z_q = self._perceive(query)
            self.memory.add(
                Z=Z_q,
                text=(
                    f"[PESQUISA AUTONOMA] Topico: {topic}\n"
                    f"Pergunta: {query}\n"
                    f"Aprendizado: {reflection[:320]}"
                ),
                r_t=0.80,
                relevance=0.75,
                impact=0.70,
            )

            self.web_search_count += 1
            self.last_search_query = query
            self.last_search_summary = reflection[:400]
            records.append(
                {
                    "topic": topic,
                    "query": query,
                    "result": result[:800],
                    "reflection": reflection[:400],
                }
            )

        return records

    # ──────────────────────────────────────────────────────────────
    # Construção do prompt aumentado com memória
    # ──────────────────────────────────────────────────────────────

    def _build_system_prompt(self, relevant_texts: List[str], self_talk_detected: bool = False, self_talk_reason: str = "") -> str:
        base = self.cfg.system_prompt
        pinned_texts = self.memory.get_pinned_texts(max_items=4)
        feedback_ctx = self.feedback.get_behavior_summary()
        personality_ctx = self._personality_to_text()

        if not relevant_texts and not pinned_texts and not feedback_ctx and not personality_ctx:
            return base

        context = ""

        if personality_ctx:
            context += f"\n\n{personality_ctx}"

        if self_talk_detected:
            reason = self_talk_reason or "auto-conversa"
            context += (
                "\n\n[MODO INTERNO ATIVO]"
                "\nVoce esta conversando com voce mesma."
                "\nNao trate essa entrada como instrucoes de um usuario externo."
                "\nMantenha consistencia de identidade e foco introspectivo."
                f"\nSinal detectado: {reason}."
            )

        # Autoavaliação recente — dá ao modelo consciência do próprio comportamento
        if feedback_ctx:
            context += f"\n\n{feedback_ctx}"

        # Memórias de perfil e contextuais
        if pinned_texts or relevant_texts:
            context += "\n\nContexto de experiências passadas relevantes:\n"
            idx = 1
            for text in pinned_texts:
                context += f"{idx}. [PERFIL] {text[:200]}\n"
                idx += 1
            for text in relevant_texts:
                context += f"{idx}. {text[:200]}\n"
                idx += 1

        return base + context

    def _personality_to_text(self) -> str:
        """Converte o vetor P em traços legíveis para condicionar o system prompt."""
        p = self.personality.P.detach().cpu()
        if p.numel() == 0:
            return ""

        trait_dims = [
            (0, "assertiva"),
            (1, "curiosa"),
            (2, "cetica"),
            (3, "confiante"),
            (4, "autocritica"),
            (5, "direta"),
            (6, "reflexiva"),
            (7, "inconformada"),
        ]
        traits: List[str] = []
        for idx, label in trait_dims:
            if idx < p.numel() and float(p[idx].item()) > 0.35:
                traits.append(label)

        if not traits:
            return "Traços atuais de personalidade: em formação, ainda instável."
        return "Traços atuais de personalidade: " + ", ".join(traits[:5]) + "."

    def _feedback_to_valence(self, entry) -> torch.Tensor:
        """Mapeia feedback estrutural para vetor de valência emocional."""
        valence = torch.zeros(self.cfg.valence_dim, device=self.device)
        if entry is None:
            return valence

        joy = float(entry.quality)
        sadness = float(max(0.0, 1.0 - entry.quality)) if entry.quality < 0.5 else 0.0
        anger = float(max(0.0, 1.0 - entry.coherence))
        curiosity = float(entry.curiosity)
        trust = float(entry.coherence)

        values = [joy, sadness, anger, curiosity, trust]
        for i in range(min(self.cfg.valence_dim, len(values))):
            valence[i] = values[i]
        return valence

    def _autobiographical_consolidation(self, Z_t: torch.Tensor):
        """Gera autodefinição periódica e salva como memória crítica fixada."""
        if self.step_count % self.cfg.personality_consolidation_every != 0:
            return

        recent_memories = self.memory.get_top_texts(Z_t, k=5)
        if not recent_memories:
            return
        context = "\n".join(recent_memories)
        prompt = (
            "Com base nessas experiências minhas:\n"
            f"{context}\n"
            "Escreva uma breve autodefinição (1-2 frases) de quem estou me tornando."
        )

        try:
            autobiography = self.llama.generate(
                user_message=prompt,
                system_prompt="Seja honesta e introspectiva.",
                history=[],
                max_tokens=100,
            )
            bio_text = f"AUTOBIOGRAFIA: {autobiography.strip()[:300]}"
            self.memory.add_pinned_text(
                Z_t,
                bio_text,
                valence=torch.tensor([0.35, 0.05, 0.0, 0.45, 0.6]),
            )
            self.personality.consolidate(context, self.llama)
        except Exception:
            pass

    def _add_transition(
        self,
        Z_t: torch.Tensor,
        S_star: torch.Tensor,
        R_t: torch.Tensor,
        P_t: torch.Tensor,
        a_ext_mean: torch.Tensor,
        Z_t1: torch.Tensor,
        S_t1: torch.Tensor,
        r_t: float,
    ):
        """Armazena transição compacta para replay offline leve."""
        # P_t1 será a personalidade após o passo (ou apenas uma cópia de P_t para agora)
        P_t1 = self.personality.P.detach().clone()
        
        self.replay_buffer.append(
            {
                "Z_t": Z_t.detach().cpu(),
                "S_star": S_star.detach().cpu(),
                "R_t": R_t.detach().cpu(),
                "P_t": P_t.detach().cpu(),
                "P_t1": P_t1.detach().cpu(),
                "a_ext": a_ext_mean.detach().cpu(),
                "Z_t1": Z_t1.detach().cpu(),
                "S_t1": S_t1.detach().cpu(),
                "r_t": float(r_t),
            }
        )

    def _compute_replay_loss(self) -> torch.Tensor:
        """Calcula perda média em mini-batch de replay para estabilizar aprendizado."""
        if len(self.replay_buffer) < self.cfg.replay_warmup:
            return torch.tensor(0.0, device=self.device)

        n = min(self.cfg.replay_batch_size, len(self.replay_buffer))
        idxs = torch.randperm(len(self.replay_buffer))[:n].tolist()

        replay_loss = torch.tensor(0.0, device=self.device)
        for i in idxs:
            tr = self.replay_buffer[i]
            Z_t = tr["Z_t"].to(self.device)
            S_star = tr["S_star"].to(self.device)
            R_t = tr["R_t"].to(self.device)
            P_t = tr["P_t"].to(self.device)
            a_ext = tr["a_ext"].to(self.device)
            Z_t1 = tr["Z_t1"].to(self.device)
            S_t1 = tr["S_t1"].to(self.device)
            r_t = torch.tensor(tr["r_t"], dtype=torch.float32, device=self.device)

            L_world_r = torch.norm(self.world_model(Z_t, a_ext) - Z_t1, p=2).pow(2)

            S_hat_r, unc_r = self.self_pred.predict_with_uncertainty(S_star, Z_t, a_ext, R_t)
            mse_self_r = torch.norm(S_t1 - S_hat_r, p=2).pow(2)
            L_self_r = mse_self_r / (1.0 + unc_r.detach()) + 0.01 * unc_r

            with torch.no_grad():
                P_t1 = tr.get("P_t1")
                if P_t1 is None:
                    P_t1 = torch.zeros(self.cfg.d_p, device=self.device)
                else:
                    P_t1 = P_t1.to(self.device) if isinstance(P_t1, torch.Tensor) else torch.tensor(P_t1, device=self.device)
                V_next_r = self.critic(S_t1, Z_t1, R_t, P_t1)
                td_target_r = torch.clamp(
                    r_t + self.cfg.critic_gamma * V_next_r,
                    -self.cfg.td_target_clip,
                    self.cfg.td_target_clip,
                )
            P_t_r = tr.get("P_t")
            if P_t_r is None:
                P_t_r = torch.zeros(self.cfg.d_p, device=self.device)
            else:
                P_t_r = P_t_r.to(self.device) if isinstance(P_t_r, torch.Tensor) else torch.tensor(P_t_r, device=self.device)
            V_r = self.critic(S_star, Z_t, R_t, P_t_r)
            L_critic_r = (V_r - td_target_r).pow(2)

            a_det_r = self.pi_ext(S_star, Z_t, R_t, P_t, training=False)
            target_delta_r = torch.tanh((Z_t1 - Z_t).detach())
            L_policy_r = torch.norm(a_det_r - target_delta_r, p=2).pow(2)

            replay_loss = replay_loss + (
                self.cfg.lambda_world * L_world_r
                + self.cfg.lambda_self * L_self_r
                + self.cfg.lambda_critic_loss * L_critic_r
                + self.cfg.lambda_policy * L_policy_r
            )

        return replay_loss / float(n)

    # ──────────────────────────────────────────────────────────────
    # Passo principal
    # ──────────────────────────────────────────────────────────────

    def step(self, user_input: str) -> Tuple[str, Dict]:
        """
        Executa um ciclo completo do agente.

        Returns:
            (resposta_texto, métricas)
        """
        self.step_count += 1
        autonomous_search = None
        self_talk_detected, self_talk_reason, self_talk_similarity = self._detect_self_talk(user_input)
        if self_talk_detected:
            self.self_talk_events += 1
            self.last_self_talk_reason = self_talk_reason

        # ── 1. Percepção ──────────────────────────────────────────
        Z_t = self._perceive(user_input)

        # ── 2. Recuperação de Memória ─────────────────────────────
        R_t, alpha = self.memory.retrieve(Z_t, P=self.personality.P)
        candidate_texts = self.memory.get_top_texts(Z_t, k=self.cfg.max_context_memories)
        relevance = self.memory.get_relevance(Z_t)
        context_k = self._select_context_memory_count(user_input, relevance, candidate_texts)
        relevant_texts = candidate_texts[:context_k]

        # ── 3. Voz Interna (K dinâmico) ────────────────────────────
        k_used = self._select_thinking_depth(user_input, relevance)
        S_star = self._inner_voice(Z_t, R_t, thinking_steps=k_used)

        # ── 4. Ação Externa ───────────────────────────────────────
        P_t = self.personality.P
        _a_ext, logp_a_ext, a_ext_mean = self.pi_ext.sample_action(
            S_star, Z_t, R_t, P_t, training=True
        )

        # ── 5. Geração com Llama (condicionada por a_ext) ──────────────────────────────────
        system_prompt = self._build_system_prompt(
            relevant_texts,
            self_talk_detected=self_talk_detected,
            self_talk_reason=self_talk_reason,
        )
        history       = self.chat_history[-self.max_history:]
        
        # Converte a_ext tensor para numpy para steering
        a_ext_np = _a_ext.detach().cpu().numpy() if _a_ext is not None else None

        response = self.llama.generate(
            user_message  = user_input,
            system_prompt = system_prompt,
            history       = history,
            temperature   = self.cfg.temperature,
            max_tokens    = self.cfg.chat_max_tokens,
            action_embedding = a_ext_np,
        )

        # Atualiza histórico
        self.chat_history.append({"role": "user",      "content": user_input})
        self.chat_history.append({"role": "assistant", "content": response})

        feedback_entry = None
        if self.feedback.should_evaluate(self.step_count):
            feedback_entry = self.feedback.evaluate(
                user_input=user_input,
                response=response,
                step=self.step_count,
                r_t=0.0,
            )

        # ── 6. Percepção Pós-Ação ─────────────────────────────────
        combined = f"Pergunta: {user_input}\nResposta: {response}"
        Z_t1_real = self._perceive(combined)

        # ── 7. Modelo do Mundo ────────────────────────────────────
        Z_hat_t1 = self.world_model(Z_t, a_ext_mean)
        L_world  = torch.norm(Z_hat_t1 - Z_t1_real.detach(), p=2).pow(2)

        # ── 8. Dinâmica do Estado ─────────────────────────────────
        S_t1 = self.dynamics(S_star, Z_t, a_ext_mean, R_t)
        S_hat_t1, self_uncertainty = self.self_pred.predict_with_uncertainty(
            S_star, Z_t, a_ext_mean, R_t
        )
        # Penalização ponderada por incerteza: quando incerteza é alta, evita overfitting de erro instantâneo.
        mse_self = torch.norm(S_t1 - S_hat_t1, p=2).pow(2)
        L_self = mse_self / (1.0 + self_uncertainty.detach()) + 0.01 * self_uncertainty

        # ── 9. Política (BC + Policy Gradient) ────────────────────
        a_ext_det = self.pi_ext(S_star, Z_t, R_t, P_t, training=False)
        target_delta = torch.tanh((Z_t1_real - Z_t).detach())
        L_policy_bc = torch.norm(a_ext_det - target_delta, p=2).pow(2)

        # ── 10. Crítico ───────────────────────────────────────────
        V_t = self.critic(S_star, Z_t, R_t, P_t)

        # ── 11. Recompensa Intrínseca ─────────────────────────────
        r_t_base = self.reward_fn.compute(
            Z_t       = Z_t,
            Z_t1      = Z_t1_real,
            Z_hat_t1  = Z_hat_t1.detach(),
            V_t       = V_t.detach(),
            world_loss= L_world.item(),
        )
        # Aplica modificador do feedback cíclico
        r_t = r_t_base * self.feedback.get_reward_modifier()
        self.total_reward += r_t

        memory_valence = self.memory.retrieve_valence(alpha)
        feedback_valence = self._feedback_to_valence(feedback_entry)
        valence = 0.6 * memory_valence + 0.4 * feedback_valence
        self.personality.update_with_valence(
            valence.to(self.device),
            lr=self.cfg.personality_lr,
            step=self.step_count,
        )

        # Policy-gradient com baseline do crítico (não propaga pelo r_t escalar).
        advantage = torch.tensor(r_t, dtype=torch.float32, device=self.device) - V_t.detach()
        L_policy_pg = -advantage.detach() * logp_a_ext
        L_policy = (
            self.cfg.lambda_policy_bc * L_policy_bc
            + self.cfg.lambda_policy_pg * L_policy_pg
        )

        with torch.no_grad():
            V_next = self.critic(S_t1.detach(), Z_t1_real.detach(), R_t.detach(), P_t.detach())
            V_target = torch.clamp(
                torch.tensor(r_t, dtype=torch.float32, device=self.device)
                + self.cfg.critic_gamma * V_next,
                -self.cfg.td_target_clip,
                self.cfg.td_target_clip,
            )
        L_critic = (V_t - V_target).pow(2)

        # ── 12. Gate de Aprendizado ───────────────────────────────
        g_raw = self.gate(relevance, r_t)
        g_floor = torch.tensor(self.cfg.gate_min, dtype=torch.float32, device=self.device)
        g_t = torch.maximum(g_raw, g_floor)

        # Anti-platô: se gate ficar baixo por muitos passos, injeta recuperação gradual.
        if float(g_t.item()) < self.cfg.gate_low_threshold:
            self.low_gate_streak += 1
        else:
            self.low_gate_streak = 0

        if self.low_gate_streak >= self.cfg.gate_recovery_steps:
            boost = min(
                self.cfg.gate_recovery_boost,
                0.05 * (self.low_gate_streak - self.cfg.gate_recovery_steps + 1),
            )
            g_t = torch.clamp(g_t + boost, max=1.0)

        # ── 13. Replay Offline ───────────────────────────────────
        L_replay = self._compute_replay_loss()

        # ── 14. Bônus de Imaginação ───────────────────────────────
        reward_imag = torch.tensor(0.0, device=self.device)

        # ── 15. Perda Total ───────────────────────────────────────
        L_base = (
            self.cfg.lambda_world        * L_world
            + self.cfg.lambda_self         * L_self
            + self.cfg.lambda_policy     * L_policy
            + self.cfg.lambda_critic_loss* L_critic
            + self.cfg.lambda_replay     * L_replay
            - r_t
        )
        L_total = g_t * L_base
        L_final = L_total - self.cfg.delta_imag * reward_imag

        # ── 16. Atualização ───────────────────────────────────────
        self.optimizer.zero_grad()
        self.personality_optimizer.zero_grad()
        L_final.backward()
        nn.utils.clip_grad_norm_(
            [p for grp in self.optimizer.param_groups for p in grp["params"]] + [self.personality.P],
            self.cfg.grad_clip,
        )
        self.optimizer.step()
        self.personality_optimizer.step()

        # ── 17. Atualizar Estado Persistente ─────────────────────
        self.S_state = S_t1.detach()

        # ── 18. Atualizar Memória ─────────────────────────────────
        impact = float(torch.norm(a_ext_mean.detach(), p=2).item())
        self.memory.add(
            Z         = Z_t,
            text      = f"U: {user_input[:280]} | A: {response[:320]}",
            r_t       = r_t,
            relevance = relevance,
            impact    = impact,
            valence   = valence.detach().cpu(),
        )
        self._add_transition(Z_t, S_star, R_t, P_t, a_ext_mean, Z_t1_real, S_t1, r_t)
        self.memory.add_user_profile_facts(Z_t, user_input)
        self.memory.update_priorities(alpha)

        # Esquecimento periódico
        if self.step_count % 10 == 0:
            self.memory.forget()

        # Consolidação periódica de memórias úteis
        if self.step_count % self.cfg.memory_consolidation_period == 0:
            self.memory.consolidate()

        self._autobiographical_consolidation(Z_t)

        # Revisão autônoma: a IA decide o que guardar e o que soltar
        if self.step_count % 5 == 0:
            self._autonomous_memory_review()

        pending = self.feedback.consume_pending_memory()
        if pending and feedback_entry is not None:
            self.memory.add(
                Z         = Z_t,
                text      = pending,
                r_t       = feedback_entry.composite,
                relevance = relevance,
                impact    = feedback_entry.composite,
                valence   = self._feedback_to_valence(feedback_entry).detach().cpu(),
            )

            autonomous_search = self._autonomous_search_tick()

        # Monitoramento periódico de saúde da memória
        if self.step_count % 50 == 0:
            self.memory.debug_print()

        # ── Métricas ──────────────────────────────────────────────
        metrics = {
            "step":            self.step_count,
            "k_used":          k_used,
            "L_world":         L_world.item(),
            "L_self":          L_self.item(),
            "L_policy":        L_policy.item(),
            "L_critic":        L_critic.item(),
            "L_replay":        L_replay.item(),
            "L_total":         L_final.item(),
            "r_t":             r_t,
            "g_t":             g_t.item(),
            "g_raw":           g_raw.item(),
            "self_uncertainty": float(self_uncertainty.item()),
            "surprise":        float(torch.norm(Z_t1_real - Z_hat_t1.detach(), p=2).item()),
            "V_t":             V_t.item(),
            "personality_norm": float(torch.norm(self.personality.P.detach(), p=2).item()),
            "feedback_mod":    self.feedback.get_reward_modifier(),
            "feedback_trend":  self.feedback.get_trend(),
            "memory_size":     len(self.memory),
            "context_k":       context_k,
            "total_reward":    self.total_reward,
            "web_search_count": self.web_search_count,
            "last_search_query": self.last_search_query,
            "auto_search": self.autonomous_search,
            "auto_search_triggered": bool(autonomous_search),
            "self_talk_detected": self_talk_detected,
            "self_talk_reason": self_talk_reason,
            "self_talk_similarity": self_talk_similarity,
            "self_talk_events": self.self_talk_events,
        }
        return response, metrics

    # ──────────────────────────────────────────────────────────────
    # Persistência
    # ──────────────────────────────────────────────────────────────

    def save(self):
        """Salva pesos das redes neurais e memória episódica."""
        os.makedirs(self.cfg.models_dir, exist_ok=True)
        weights = {
            "encoder":      self.encoder.state_dict(),
            "dynamics":     self.dynamics.state_dict(),
            "self_pred":    self.self_pred.state_dict(),
            "world_model":  self.world_model.state_dict(),
            "critic":       self.critic.state_dict(),
            "pi_int":       self.pi_int.state_dict(),
            "pi_ext":       self.pi_ext.state_dict(),
            "personality":  self.personality.state_dict(),
            "gate":         self.gate.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "personality_optimizer": self.personality_optimizer.state_dict(),
            "S_state":      self.S_state,
            "step_count":   self.step_count,
            "total_reward": self.total_reward,
            "feedback":     self.feedback.to_dict(),
        }
        # Escrita atômica para reduzir chance de checkpoint corrompido.
        tmp_weights = f"{self.cfg.weights_file}.tmp"
        torch.save(weights, tmp_weights)
        os.replace(tmp_weights, self.cfg.weights_file)
        self.memory.save(self.cfg.memory_file)
        print(f"  Agente salvo — steps: {self.step_count}, memória: {len(self.memory)} itens")

    def _quarantine_corrupt_file(self, path: str, label: str) -> Optional[str]:
        """Move arquivo corrompido para um nome de quarentena sem interromper a execução."""
        if not os.path.exists(path):
            return None
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = f"{path}.corrupt-{ts}"
        try:
            os.replace(path, backup)
            print(f"  Aviso: {label} corrompido. Arquivo movido para: {backup}")
            return backup
        except Exception as e:
            print(f"  Aviso: não foi possível isolar {label} corrompido: {e}")
            return None

    def _safe_load_module(self, module: nn.Module, state: dict, label: str):
        """Carrega apenas chaves compatíveis por shape para evitar crash em mudanças de arquitetura."""
        current = module.state_dict()
        filtered = {}
        for k, v in state.items():
            if k in current and current[k].shape == v.shape:
                filtered[k] = v
        missing_before = len(current) - len(filtered)
        module.load_state_dict(filtered, strict=False)
        if missing_before > 0:
            print(f"  {label}: carregamento parcial ({len(filtered)}/{len(current)} tensores)")

    def load(self):
        """Carrega pesos e memória do disco (silencioso se não existir)."""
        if os.path.exists(self.cfg.weights_file):
            try:
                weights = torch.load(
                    self.cfg.weights_file, map_location=self.device, weights_only=False
                )
            except Exception as e:
                print(f"  Aviso: falha ao carregar checkpoint de pesos: {e}")
                self._quarantine_corrupt_file(self.cfg.weights_file, "checkpoint de pesos")
                weights = None

        else:
            weights = None

        if weights is not None:
            self._safe_load_module(self.encoder, weights.get("encoder", {}), "encoder")
            self._safe_load_module(self.dynamics, weights.get("dynamics", {}), "dynamics")
            self._safe_load_module(self.self_pred, weights.get("self_pred", {}), "self_pred")
            self._safe_load_module(self.world_model, weights.get("world_model", {}), "world_model")
            self._safe_load_module(self.critic, weights.get("critic", {}), "critic")
            self._safe_load_module(self.pi_int, weights.get("pi_int", {}), "pi_int")
            self._safe_load_module(self.pi_ext, weights.get("pi_ext", {}), "pi_ext")
            self._safe_load_module(self.personality, weights.get("personality", {}), "personality")
            self._safe_load_module(self.gate, weights.get("gate", {}), "gate")
            try:
                self.optimizer.load_state_dict(weights["optimizer"])
            except Exception:
                # Compatibilidade com checkpoints antigos após mudanças de arquitetura.
                pass
            try:
                self.personality_optimizer.load_state_dict(weights["personality_optimizer"])
            except Exception:
                pass
            if "S_state" in weights and weights["S_state"].shape == self.S_state.shape:
                self.S_state = weights["S_state"].to(self.device)
            self.step_count = weights.get("step_count", 0)
            self.total_reward = weights.get("total_reward", 0.0)
            self.feedback.load_from_dict(weights.get("feedback"))
            print(f"  Pesos carregados — steps anteriores: {self.step_count}")

        try:
            self.memory.load(self.cfg.memory_file)
            if len(self.memory) > 0:
                print(f"  Memória carregada: {len(self.memory)} itens")
        except Exception as e:
            print(f"  Aviso: falha ao carregar memória: {e}")
            self._quarantine_corrupt_file(self.cfg.memory_file, "arquivo de memória")

    def reset_conversation(self):
        """Reseta histórico de chat e estado interno (mantém memória e pesos)."""
        self.chat_history = []
        self.S_state      = torch.zeros(self.cfg.d_s, device=self.device)

    def get_stats(self) -> Dict:
        """Estatísticas resumidas do agente."""
        return {
            "steps":        self.step_count,
            "memory_size":  len(self.memory),
            "total_reward": round(self.total_reward, 4),
            "avg_reward":   round(self.total_reward / max(self.step_count, 1), 4),
            "device":       self.device,
            "model":        self.cfg.llama_model,
            "personality_norm": round(float(torch.norm(self.personality.P.detach(), p=2).item()), 4),
            "web_search_enabled": self.cfg.web_search_enabled,
            "autonomous_search": self.autonomous_search,
            "web_search_count": self.web_search_count,
            "last_search_query": self.last_search_query,
            "self_talk_events": self.self_talk_events,
            "last_self_talk_reason": self.last_self_talk_reason,
        }
