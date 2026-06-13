"""
Memória Episódica Priorizada com Gerenciamento Inteligente
===========================================================
Armazena experiências passadas com categorização de importância.
Recuperação via atenção softmax ponderada por similaridade + prioridade + idade.

CAMADAS DE MEMÓRIA:
    CRÍTICA     : λ_decay_crit << λ_decay_normal  [dados-chave: nomes, contexto]
    IMPORTANTE  : λ_decay_imp << λ_decay_normal   [aprendizados]
    CONTEXTUAL  : λ_decay_normal                  [experiências temporárias]

MECANISMOS:
    1. Proteger crítica: taxa de esquecimento 80% menor
    2. Renovação automática: resgatar crítica periodicamente (sem uso)
    3. Consolidação: garantir mínimo de retenção
    4. Recência ponderada: idade afeta score, mas crítica resiste

Fórmulas:
    λ_decay(m) = λ_norm · (1 - importance_mult_m)
    score_m    = sim(Z_t, Z_m) + λ_p·p_m - λ_age·age(m)
    renewal    = p_m * (1 + renewal_factor) se m ∈ CRÍTICA e time % renew_period == 0
"""

import os
import pickle
import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from enum import Enum

from config import Config


class MemoryImportance(Enum):
    """Categorias de importância com proteção diferenciada."""
    CRITICAL  = 3    # Nome, identidade, contexto central
    IMPORTANT = 2    # Aprendizados, relações
    CONTEXTUAL= 1    # Experiências temporárias


class EpisodicMemory:
    """Memória Episódica com Gerenciamento Inteligente de Esquecimento."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embeddings: List[torch.Tensor] = []      # [d_e] cada
        self.texts: List[str] = []
        self.priorities: List[float] = []
        self.importance: List[MemoryImportance] = []  # Camada de cada item
        self.valences: List[torch.Tensor] = []        # [valence_dim] por item
        self.timestamps: List[int] = []               # Quando foi adicionado
        self.access_count: List[int] = []             # Vezes acessado
        self.is_pinned: List[bool] = []               # Itens fixos (perfil/identidade)
        # Autonomia de memória: multiplicador de decay escolhido pela própria IA.
        # None  = automático (usa decay_multipliers por importância)
        # 0.05  = ela quer guardar (quase imortal)
        # 3.0   = ela quer soltar (decai rápido)
        self.agent_decay_override: List[Optional[float]] = []
        self.time_step = 0                            # Contador global
        
        # Parâmetros de proteção
        self.decay_multipliers = {
            MemoryImportance.CRITICAL:   0.2,    # 80% de proteção
            MemoryImportance.IMPORTANT:  0.5,    # 50% de proteção
            MemoryImportance.CONTEXTUAL: 1.0,    # Taxa normal
        }
        self.renewal_period = 20              # A cada 20 passos, renovar crítica
        self.renewal_boost = 0.5              # +50% ao renovar
        self.max_priority = 25.0              # limite superior para evitar explosão numérica
        self.pinned_bonus = 1.5               # bônus na recuperação para fatos de perfil

    def __len__(self) -> int:
        return len(self.embeddings)
    
    def _auto_renew_critical(self):
        """Renova automaticamente itens CRÍTICOS para evitar esquecimento."""
        if self.time_step % self.renewal_period != 0:
            return
        
        for i, imp in enumerate(self.importance):
            if imp == MemoryImportance.CRITICAL or self.is_pinned[i]:
                self.priorities[i] *= (1.0 + self.renewal_boost)
                self.priorities[i] = min(self.priorities[i], self.max_priority)
                self.access_count[i] += 1

    def _extract_profile_facts(self, user_text: str) -> List[str]:
        """Extrai fatos pessoais estáveis para memória de longo prazo."""
        text = user_text.strip()
        lower = text.lower()
        facts: List[str] = []

        patterns = [
            (r"(?:meu nome e|meu nome é|i am|i'm)\s+([a-zA-ZÀ-ÿ'\- ]{2,40})", "nome"),
            (r"(?:eu moro em|i live in)\s+([a-zA-ZÀ-ÿ'\- ]{2,60})", "local"),
            (r"(?:meu objetivo e|meu objetivo é|my goal is)\s+(.{3,120})", "objetivo"),
            (r"(?:eu gosto de|i like)\s+(.{3,120})", "gosto"),
            (r"(?:eu sou|i am)\s+(.{3,120})", "identidade"),
        ]

        for pattern, tag in patterns:
            m = re.search(pattern, lower, flags=re.IGNORECASE)
            if not m:
                continue
            raw_value = m.group(1).strip(" .,!?:;")
            if not raw_value:
                continue
            if tag == "nome":
                facts.append(f"FACT: nome do usuario = {raw_value}")
            elif tag == "local":
                facts.append(f"FACT: local do usuario = {raw_value}")
            elif tag == "objetivo":
                facts.append(f"FACT: objetivo do usuario = {raw_value}")
            elif tag == "gosto":
                facts.append(f"FACT: preferencias do usuario = {raw_value}")
            elif tag == "identidade":
                facts.append(f"FACT: identidade declarada do usuario = {raw_value}")

        # Remove duplicados preservando ordem
        seen = set()
        unique_facts: List[str] = []
        for fact in facts:
            if fact in seen:
                continue
            seen.add(fact)
            unique_facts.append(fact)
        return unique_facts

    def _replace_index(
        self,
        idx: int,
        Z: torch.Tensor,
        text: str,
        priority: float,
        importance: MemoryImportance,
        pinned: bool,
        valence: Optional[torch.Tensor] = None,
    ):
        self.embeddings[idx] = Z.detach().cpu()
        self.texts[idx] = text
        self.priorities[idx] = min(priority, self.max_priority)
        self.importance[idx] = importance
        if valence is None:
            valence = torch.zeros(self.cfg.valence_dim)
        self.valences[idx] = valence.detach().cpu().view(-1)[: self.cfg.valence_dim]
        self.timestamps[idx] = self.time_step
        self.access_count[idx] = 0
        self.is_pinned[idx] = pinned
        self.agent_decay_override[idx] = None  # Reseta escolha da IA ao substituir

    def _append_item(
        self,
        Z: torch.Tensor,
        text: str,
        priority: float,
        importance: MemoryImportance,
        pinned: bool,
        valence: Optional[torch.Tensor] = None,
    ):
        self.embeddings.append(Z.detach().cpu())
        self.texts.append(text)
        self.priorities.append(min(priority, self.max_priority))
        self.importance.append(importance)
        if valence is None:
            valence = torch.zeros(self.cfg.valence_dim)
        self.valences.append(valence.detach().cpu().view(-1)[: self.cfg.valence_dim])
        self.timestamps.append(self.time_step)
        self.access_count.append(0)
        self.is_pinned.append(pinned)
        self.agent_decay_override.append(None)

    def add_user_profile_facts(self, Z: torch.Tensor, user_text: str):
        """Insere/atualiza fatos pessoais estáveis em memória fixada (pinned)."""
        facts = self._extract_profile_facts(user_text)
        if not facts:
            return

        for fact in facts:
            # Se já existe, só reforça muito
            try:
                idx = self.texts.index(fact)
                self.priorities[idx] = min(max(self.priorities[idx], 8.0) + 1.0, self.max_priority)
                self.importance[idx] = MemoryImportance.CRITICAL
                self.is_pinned[idx] = True
                self.timestamps[idx] = self.time_step
                continue
            except ValueError:
                pass

            # Se não existe, adiciona como item protegido
            if len(self.embeddings) >= self.cfg.memory_size:
                # Nunca remove item pinned para inserir outro
                candidates = [
                    i for i, pin in enumerate(self.is_pinned)
                    if not pin
                ]
                if not candidates:
                    # Se tudo estiver pinned, apenas reforça o melhor item crítico existente
                    idx = int(np.argmax(self.priorities))
                    self.priorities[idx] = min(self.priorities[idx] + 0.5, self.max_priority)
                    continue
                weighted = [
                    self.priorities[i] * (1.0 / self.decay_multipliers[self.importance[i]])
                    for i in candidates
                ]
                min_local = int(np.argmin(weighted))
                min_idx = candidates[min_local]
                self._replace_index(
                    min_idx,
                    Z,
                    fact,
                    priority=10.0,
                    importance=MemoryImportance.CRITICAL,
                    pinned=True,
                    valence=torch.tensor([0.3, 0.0, 0.0, 0.4, 0.6]),
                )
            else:
                self._append_item(
                    Z,
                    fact,
                    priority=10.0,
                    importance=MemoryImportance.CRITICAL,
                    pinned=True,
                    valence=torch.tensor([0.3, 0.0, 0.0, 0.4, 0.6]),
                )

    # ──────────────────────────────────────────────────────────────
    # Recuperação
    # ──────────────────────────────────────────────────────────────

    def retrieve(self, Z_t: torch.Tensor, P: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recupera R_t via atenção priorizada + renovação automática de crítica.

        Args:
            Z_t: [d_e]
        Returns:
            R_t   : [d_e] — vetor de contexto recuperado
            alpha : [|M|]  — pesos de atenção (tensor vazio se memória vazia)
        """
        # Aplica renovação automática de itens críticos
        self._auto_renew_critical()
        self.time_step += 1
        
        if len(self.embeddings) == 0:
            return torch.zeros(self.cfg.d_e, device=Z_t.device), torch.tensor([])

        Z_mem = torch.stack(self.embeddings).to(Z_t.device)          # [M, d_e]
        prio  = torch.tensor(self.priorities, dtype=torch.float32,
                             device=Z_t.device)                       # [M]

        # Similaridade cosseno
        Z_t_n   = F.normalize(Z_t.unsqueeze(0), dim=-1)              # [1, d_e]
        Z_mem_n = F.normalize(Z_mem, dim=-1)                         # [M, d_e]
        sims    = (Z_t_n * Z_mem_n).sum(dim=-1)                      # [M]

        # Fator de idade (recência): reduz influência de memórias muito antigas
        ages = torch.tensor([self.time_step - ts for ts in self.timestamps],
                           dtype=torch.float32, device=Z_t.device)
        age_penalty = 0.05 * (ages / (1.0 + ages.max()))  # Normalizado

        # Pontuação combinada: similaridade + prioridade - penalidade de idade
        pin_bonus = torch.tensor(
            [self.pinned_bonus if pin else 0.0 for pin in self.is_pinned],
            dtype=torch.float32,
            device=Z_t.device,
        )

        valence_scores = torch.zeros_like(sims)
        if P is not None and len(self.valences) == len(self.embeddings):
            val_tensor = torch.stack(self.valences).to(Z_t.device)  # [M, d_val]
            p_vec = P.detach().to(Z_t.device).view(-1)
            if p_vec.shape[0] >= self.cfg.valence_dim:
                p_affect = torch.tanh(p_vec[: self.cfg.valence_dim])
            else:
                p_affect = torch.zeros(self.cfg.valence_dim, device=Z_t.device)
                p_affect[: p_vec.shape[0]] = torch.tanh(p_vec)
            valence_scores = (val_tensor * p_affect.unsqueeze(0)).sum(dim=-1)

        scores = (
            sims
            + self.cfg.lambda_p * prio
            + pin_bonus
            + self.cfg.personality_influence_weight * valence_scores
            - age_penalty
        )  # [M]

        # Atenção e agregação
        alpha = F.softmax(scores, dim=0)                             # [M]
        R_t   = (alpha.unsqueeze(-1) * Z_mem).sum(dim=0)            # [d_e]

        return R_t, alpha

    def retrieve_valence(self, alpha: torch.Tensor) -> torch.Tensor:
        """Retorna valência média ponderada por atenção alpha."""
        if len(self.valences) == 0:
            return torch.zeros(self.cfg.valence_dim, device=alpha.device)
        valences_tensor = torch.stack(self.valences).to(alpha.device)
        return (alpha.unsqueeze(-1) * valences_tensor).sum(dim=0)

    def get_relevance(self, Z_t: torch.Tensor) -> float:
        """relevance_t = (1/|M|) · Σ sim(Z_t, Z_m)"""
        if len(self.embeddings) == 0:
            return 0.0
        Z_mem   = torch.stack(self.embeddings).to(Z_t.device)
        Z_t_n   = F.normalize(Z_t.unsqueeze(0), dim=-1)
        Z_mem_n = F.normalize(Z_mem, dim=-1)
        sims    = (Z_t_n * Z_mem_n).sum(dim=-1)
        return float(sims.mean().item())

    def get_top_texts(self, Z_t: torch.Tensor, k: int = 3) -> List[str]:
        """Retorna os k textos mais relevantes para Z_t com proteção de crítica."""
        if len(self.embeddings) == 0:
            return []
        Z_mem  = torch.stack(self.embeddings).to(Z_t.device)
        prio   = torch.tensor(self.priorities, dtype=torch.float32, device=Z_t.device)
        Z_t_n  = F.normalize(Z_t.unsqueeze(0), dim=-1)
        Z_mn   = F.normalize(Z_mem, dim=-1)
        sims   = (Z_t_n * Z_mn).sum(dim=-1)
        
        # Penalty menor para crítica (resiste mais a idade)
        ages = torch.tensor([self.time_step - ts for ts in self.timestamps],
                           dtype=torch.float32, device=Z_t.device)
        age_penalty = torch.tensor([
            (0.01 if imp == MemoryImportance.CRITICAL else 0.05) * (age / (1.0 + ages.max()))
            for imp, age in zip(self.importance, ages)
        ], device=Z_t.device)
        pin_bonus = torch.tensor(
            [self.pinned_bonus if pin else 0.0 for pin in self.is_pinned],
            dtype=torch.float32,
            device=Z_t.device,
        )
        
        scores = sims + self.cfg.lambda_p * prio + pin_bonus - age_penalty
        k      = min(k, len(self.texts))
        top_i  = scores.topk(k).indices.tolist()
        return [self.texts[i] for i in top_i]

    def get_pinned_texts(self, max_items: int = 4) -> List[str]:
        """Retorna fatos pinned para contexto estável (nome, identidade, preferências)."""
        items = [
            (txt, prio)
            for txt, prio, pin in zip(self.texts, self.priorities, self.is_pinned)
            if pin
        ]
        if not items:
            return []
        items.sort(key=lambda x: x[1], reverse=True)
        return [txt for txt, _ in items[:max_items]]

    # ──────────────────────────────────────────────────────────────
    # Adição e atualização
    # ──────────────────────────────────────────────────────────────

    def _detect_importance(self, text: str, relevance: float, impact: float) -> MemoryImportance:
        """Detecta importância de forma inteligente baseado no conteúdo e score."""
        # Palavras-chave indicam crítica
        critical_keywords = [
            "nome", "name", "identidade", "identity", "contexto", "context",
            "objetivo", "goal", "objetivo principal", "main goal",
            "quem é", "who is", "sou", "i am", "você é", "you are"
        ]
        
        text_lower = text.lower()
        is_critical_topic = any(kw in text_lower for kw in critical_keywords)
        
        # Pontuação de importância: impacto alto + (tema crítico ou alta relevância)
        importance_score = impact + (0.5 * relevance)
        
        if is_critical_topic or importance_score > 0.7:
            return MemoryImportance.CRITICAL
        elif importance_score > 0.4:
            return MemoryImportance.IMPORTANT
        else:
            return MemoryImportance.CONTEXTUAL

    def add(
        self,
        Z: torch.Tensor,
        text: str,
        r_t: float,
        relevance: float,
        impact: float,
        valence: Optional[torch.Tensor] = None,
    ):
        """
        Adiciona nova experiência com proteção adaptativa baseada em importância.
        
        p_new = (w_r·relevance + w_c·r_t + w_i·impact) * importance_boost
        """
        # Calcula prioridade base
        priority = (
            self.cfg.w_r * max(relevance, 0.0)
            + self.cfg.w_c * max(r_t, 0.0)
            + self.cfg.w_i * max(impact, 0.0)
        )
        
        # Detecta e reforça itens críticos
        importance = self._detect_importance(text, relevance, impact)
        priority_boost = {
            MemoryImportance.CRITICAL:   2.0,      # 2x reforço
            MemoryImportance.IMPORTANT:  1.5,      # 1.5x reforço
            MemoryImportance.CONTEXTUAL: 1.0,      # Normal
        }
        priority *= priority_boost[importance]
        priority = max(priority, self.cfg.epsilon_forget * 2)

        if len(self.embeddings) >= self.cfg.memory_size:
            # Substitui item menos importante (ponderado por importância)
            candidates = [i for i, pin in enumerate(self.is_pinned) if not pin]
            if not candidates:
                return
            weighted_priorities = [
                self.priorities[i] * (1.0 / self.decay_multipliers[self.importance[i]])
                for i in candidates
            ]
            min_local = int(np.argmin(weighted_priorities))
            min_idx = candidates[min_local]
            self._replace_index(min_idx, Z, text, priority, importance, pinned=False, valence=valence)
        else:
            self._append_item(Z, text, priority, importance, pinned=False, valence=valence)

    def add_pinned_text(self, Z: torch.Tensor, text: str, valence: Optional[torch.Tensor] = None):
        """Adiciona texto autobiográfico/factual como memória crítica fixada."""
        if not text.strip():
            return
        try:
            idx = self.texts.index(text)
            self.priorities[idx] = min(max(self.priorities[idx], 10.0) + 0.5, self.max_priority)
            self.importance[idx] = MemoryImportance.CRITICAL
            self.is_pinned[idx] = True
            if valence is not None:
                self.valences[idx] = valence.detach().cpu().view(-1)[: self.cfg.valence_dim]
            return
        except ValueError:
            pass

        if len(self.embeddings) >= self.cfg.memory_size:
            candidates = [i for i, pin in enumerate(self.is_pinned) if not pin]
            if not candidates:
                return
            weighted = [
                self.priorities[i] * (1.0 / self.decay_multipliers[self.importance[i]])
                for i in candidates
            ]
            min_idx = candidates[int(np.argmin(weighted))]
            self._replace_index(
                min_idx,
                Z,
                text,
                priority=10.0,
                importance=MemoryImportance.CRITICAL,
                pinned=True,
                valence=valence,
            )
        else:
            self._append_item(
                Z,
                text,
                priority=10.0,
                importance=MemoryImportance.CRITICAL,
                pinned=True,
                valence=valence,
            )

    def update_priorities(self, alpha: torch.Tensor):
        """
        Decaimento adaptativo por camada + reforço por uso.
        
        Crítica: λ_decay * 0.2  (protegida 80%)
        Importante: λ_decay * 0.5 (protegida 50%)
        Contextual: λ_decay * 1.0 (taxa normal)
        
        p_m ← p_m · exp(−λ_decay * decay_multiplier) + η · α_m
        """
        if len(self.priorities) == 0:
            return
            
        alpha_np = (
            alpha.detach().cpu().numpy()
            if len(alpha) > 0
            else np.zeros(len(self.priorities))
        )
        
        for i in range(len(self.priorities)):
            # Decay adaptativo:
            # 1. Base: multiplicador por camada de importância
            # 2. Override: se a IA escolheu explicitamente guardar ou soltar, prevalece
            # 3. Bônus de uso: mais acessado = decay mais lento (acesso_count reduz decay)
            base_mult = self.decay_multipliers[self.importance[i]]
            if self.agent_decay_override[i] is not None:
                # Override da IA prevalece sobre a camada
                decay_mult = float(self.agent_decay_override[i])
            else:
                decay_mult = base_mult

            # Uso frequente desacelera o decay proporcionalmente (máx 60% de redução)
            use_factor = max(0.4, 1.0 - 0.06 * min(self.access_count[i], 10))
            self.priorities[i] *= np.exp(-self.cfg.lambda_decay * decay_mult * use_factor)

            # Reforço por uso (acesso)
            if i < len(alpha_np):
                self.priorities[i] += self.cfg.eta_memory * float(alpha_np[i])
                self.access_count[i] += 1

            self.priorities[i] = min(self.priorities[i], self.max_priority)

    def forget(self):
        """
        Esquecimento seletivo:
        - modo suave (default): rebaixa itens abaixo do limiar em vez de apagar
        - modo legado: remove itens abaixo do limiar

        Crítica: limiar_crit = ε_forget * 0.1  (quase imortal)
        Importante: limiar_imp = ε_forget * 0.35
        Contextual: limiar = ε_forget
        """
        keep = []
        to_soft_demote = []
        for i, (p, imp) in enumerate(zip(self.priorities, self.importance)):
            if self.is_pinned[i]:
                keep.append(i)
                continue

            # Se a IA escolheu guardar esta memória, usa limiar quase zero
            override = self.agent_decay_override[i]
            if override is not None and override <= 0.1:  # marcada como "guardar"
                threshold = self.cfg.epsilon_forget * 0.02
            else:
                threshold = {
                    MemoryImportance.CRITICAL:   self.cfg.epsilon_forget * 0.1,
                    MemoryImportance.IMPORTANT:  self.cfg.epsilon_forget * 0.35,
                    MemoryImportance.CONTEXTUAL: self.cfg.epsilon_forget,
                }[imp]

            if p >= threshold:
                keep.append(i)
                continue

            if self.cfg.soft_forget_enabled:
                to_soft_demote.append(i)

        # Esquecimento suave: não apaga, só enfraquece para manter rastro de longo prazo.
        for i in to_soft_demote:
            self.importance[i] = MemoryImportance.CONTEXTUAL
            self.priorities[i] = max(self.priorities[i], self.cfg.soft_forget_priority_floor)
            # Mantém tendência de "soltar" sem zerar a lembrança.
            override = self.agent_decay_override[i]
            if override is None or override < self.cfg.soft_forget_release_decay:
                self.agent_decay_override[i] = self.cfg.soft_forget_release_decay
            keep.append(i)

        self.embeddings         = [self.embeddings[i]         for i in keep]
        self.texts              = [self.texts[i]              for i in keep]
        self.priorities         = [self.priorities[i]         for i in keep]
        self.importance         = [self.importance[i]         for i in keep]
        self.valences           = [self.valences[i]           for i in keep]
        self.timestamps         = [self.timestamps[i]         for i in keep]
        self.access_count       = [self.access_count[i]       for i in keep]
        self.is_pinned          = [self.is_pinned[i]          for i in keep]
        self.agent_decay_override = [self.agent_decay_override[i] for i in keep]

    # ──────────────────────────────────────────────────────────────
    # Autonomia de Memória — a IA decide o que guardar ou soltar
    # ──────────────────────────────────────────────────────────────

    def protect_memory(self, idx: int):
        """IA decide guardar esta memória: decay cai para 5% do normal."""
        if 0 <= idx < len(self.agent_decay_override):
            self.agent_decay_override[idx] = 0.05
            # Reforça prioridade imediatamente
            self.priorities[idx] = min(self.priorities[idx] * 1.5 + 1.0, self.max_priority)

    def release_memory(self, idx: int):
        """IA decide soltar esta memória: perde importância sem apagar de forma brusca."""
        if 0 <= idx < len(self.agent_decay_override):
            # Não solta pinned nem críticos de perfil
            if not self.is_pinned[idx]:
                self.importance[idx] = MemoryImportance.CONTEXTUAL
                self.agent_decay_override[idx] = max(
                    self.cfg.soft_forget_release_decay,
                    1.2,
                )
                # Reduz, mas preserva um rastro mínimo para não apagar 100%.
                self.priorities[idx] = max(
                    self.priorities[idx] * 0.7,
                    self.cfg.soft_forget_priority_floor,
                )

    def get_review_candidates(self, k: int = 8) -> List[Tuple[int, str, float, str]]:
        """
        Retorna candidatos para revisão autônoma da IA.
        Exclui itens pinned (perfil/identidade) — esses não são negociáveis.
        Retorna: lista de (idx, text, priority, status)
        onde status = 'guardar' | 'soltar' | 'neutro'
        """
        candidates = [
            (i, self.texts[i], self.priorities[i], self.importance[i])
            for i in range(len(self.texts))
            if not self.is_pinned[i]
        ]
        if not candidates:
            return []
        # Ordena por prioridade desc para mostrar as mais relevantes
        candidates.sort(key=lambda x: x[2], reverse=True)
        result = []
        for idx, text, prio, imp in candidates[:k]:
            override = self.agent_decay_override[idx]
            if override is not None and override <= 0.1:
                status = "guardar"
            elif override is not None and override >= 2.0:
                status = "soltar"
            else:
                status = "neutro"
            result.append((idx, text, prio, status))
        return result

    def get_agent_choice_summary(self) -> dict:
        """Quantos itens a IA escolheu guardar, soltar ou deixou neutro."""
        guardar = sum(1 for o in self.agent_decay_override if o is not None and o <= 0.1)
        soltar  = sum(1 for o in self.agent_decay_override if o is not None and o >= 2.0)
        neutro  = len(self.agent_decay_override) - guardar - soltar
        return {"guardar": guardar, "soltar": soltar, "neutro": neutro}

    def consolidate(self):
        """Consolida memórias úteis para reduzir esquecimento de longo prazo."""
        if len(self.embeddings) == 0:
            return

        for i, imp in enumerate(self.importance):
            age = self.time_step - self.timestamps[i]
            if self.is_pinned[i]:
                self.priorities[i] = min(max(self.priorities[i], 10.0), self.max_priority)
                continue

            # Memórias importantes e frequentemente acessadas ganham reforço periódico.
            if imp in (MemoryImportance.CRITICAL, MemoryImportance.IMPORTANT):
                if self.access_count[i] >= 2:
                    boost = 0.35 if imp == MemoryImportance.CRITICAL else 0.20
                    self.priorities[i] = min(self.priorities[i] + boost, self.max_priority)

                # Se é antiga e ainda relevante, evita cair para esquecimento acidental.
                if age > 80 and self.priorities[i] < 0.25:
                    floor = 0.25 if imp == MemoryImportance.CRITICAL else 0.15
                    self.priorities[i] = max(self.priorities[i], floor)

    def get_memory_stats(self) -> dict:
        """Retorna estatísticas detalhadas sobre saúde da memória."""
        if len(self.embeddings) == 0:
            return {"total": 0, "by_importance": {}, "summary": "Memória vazia"}
        
        stats_by_importance = {}
        for imp in MemoryImportance:
            count = sum(1 for i in self.importance if i == imp)
            if count > 0:
                priorities = [p for p, i in zip(self.priorities, self.importance) if i == imp]
                stats_by_importance[imp.name] = {
                    "count": count,
                    "avg_priority": float(np.mean(priorities)),
                    "min_priority": float(np.min(priorities)),
                    "max_priority": float(np.max(priorities)),
                }
        
        critical_at_risk = sum(
            1 for p, imp in zip(self.priorities, self.importance)
            if imp == MemoryImportance.CRITICAL and p < self.cfg.epsilon_forget * 0.5
        )
        
        return {
            "total": len(self.embeddings),
            "by_importance": stats_by_importance,
            "critical_at_risk": critical_at_risk,
            "pinned_count": sum(1 for pin in self.is_pinned if pin),
            "time_step": self.time_step,
            "next_renewal": self.renewal_period - (self.time_step % self.renewal_period),
        }
    
    def debug_print(self):
        """Imprime diagnóstico da memória em linguagem natural."""
        stats = self.get_memory_stats()
        print("\n" + "="*60)
        print("DIAGNÓSTICO DA MEMÓRIA")
        print("="*60)
        
        if stats["total"] == 0:
            print("⚠️  Memória vazia - nenhum item armazenado")
            return
        
        print(f"Total de itens: {stats['total']}")
        print(f"Itens fixos (pinned): {stats['pinned_count']}")
        print(f"Passo atual: {stats['time_step']}")
        print(f"Renovação crítica em: {stats['next_renewal']} passos\n")
        
        for imp_name, imp_stats in stats["by_importance"].items():
            print(f"📌 {imp_name}:")
            print(f"   - Itens: {imp_stats['count']}")
            print(f"   - Prioridade média: {imp_stats['avg_priority']:.3f}")
            print(f"   - Intervalo: [{imp_stats['min_priority']:.3f}, {imp_stats['max_priority']:.3f}]")
        
        if stats["critical_at_risk"] > 0:
            print(f"\n⚠️  {stats['critical_at_risk']} itens CRÍTICOS em risco de esquecimento!")

    # ──────────────────────────────────────────────────────────────
    # Persistência
    # ──────────────────────────────────────────────────────────────

    def save(self, path: str):
        """Persiste memória completa em disco."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "embeddings": [e.numpy() for e in self.embeddings],
            "texts":      self.texts,
            "priorities": self.priorities,
            "importance": [imp.name for imp in self.importance],
            "valences":   [v.numpy() for v in self.valences],
            "timestamps": self.timestamps,
            "access_count": self.access_count,
            "is_pinned": self.is_pinned,
            "agent_decay_override": self.agent_decay_override,
            "time_step": self.time_step,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        """Carrega memória do disco com compatibilidade retroativa."""
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.embeddings = [torch.from_numpy(e) for e in data["embeddings"]]
        self.texts      = data["texts"]
        self.priorities = data["priorities"]
        self.time_step  = data.get("time_step", 0)
        
        # Compatibilidade: se não tiver importância, defini como CONTEXTUAL
        if "importance" in data:
            self.importance = [MemoryImportance[imp_name] for imp_name in data["importance"]]
        else:
            self.importance = [MemoryImportance.CONTEXTUAL] * len(self.texts)

        if "valences" in data:
            self.valences = [torch.from_numpy(v) for v in data["valences"]]
        else:
            self.valences = [torch.zeros(self.cfg.valence_dim) for _ in self.texts]

        if len(self.valences) < len(self.texts):
            self.valences.extend(
                [torch.zeros(self.cfg.valence_dim) for _ in range(len(self.texts) - len(self.valences))]
            )
        elif len(self.valences) > len(self.texts):
            self.valences = self.valences[: len(self.texts)]
        
        self.timestamps = data.get("timestamps", [0] * len(self.texts))
        self.access_count = data.get("access_count", [0] * len(self.texts))
        self.is_pinned = data.get("is_pinned", [False] * len(self.texts))
        self.agent_decay_override = data.get("agent_decay_override", [None] * len(self.texts))
