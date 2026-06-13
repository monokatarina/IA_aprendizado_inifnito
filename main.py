#!/usr/bin/env python3
"""
EcoMental — A IA que pensa, lembra e decide o que aprender
============================================================
Ponto de entrada principal do sistema.

Uso:
    python main.py
"""

import sys
import os
import atexit
import signal
from types import FrameType
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from llama_bridge import LlamaBridge
from brain.agent import CentralAgent


def _safe_save(agent: CentralAgent, reason: str = ""):
    """Salva estado sem interromper o encerramento do processo."""
    try:
        suffix = f" ({reason})" if reason else ""
        print(f"\nSalvando estado do agente{suffix}...")
        agent.save()
    except Exception as e:
        print(f"\nAviso: falha ao salvar estado: {e}")


def _install_exit_handlers(agent: CentralAgent):
    """Registra salvamento automático para saídas normais e sinais do SO."""
    atexit.register(lambda: _safe_save(agent, "encerramento"))

    def _signal_handler(signum: int, _frame: Optional[FrameType]):
        _safe_save(agent, f"sinal {signum}")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)


def setup(cfg: Config):
    """Inicializa todos os componentes e valida o ambiente."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  EcoMental — A IA que pensa, lembra e decide o que aprender")
    print(sep)

    # ── Verifica Ollama ────────────────────────────────────────
    print(f"\n[1/3] Conectando ao Ollama ({cfg.ollama_url})...")
    llama = LlamaBridge(cfg)

    if not llama.is_available():
        print("  ERRO: Ollama não está rodando!")
        print("  Inicie com : ollama serve")
        sys.exit(1)

    models = llama.list_models()
    print(f"  Modelos disponíveis: {', '.join(models) if models else 'nenhum'}")

    if not models:
        print("\n  ERRO: Nenhum modelo instalado no Ollama.")
        print("  Execute: ollama pull qwen3.5b")
        sys.exit(1)

    # Usa o modelo configurado se disponível, senão usa o primeiro disponível
    model_found = next((m for m in models if cfg.llama_model in m), None)
    if model_found:
        cfg.llama_model = model_found
    else:
        cfg.llama_model = models[0]
        print(f"  Modelo '{cfg.llama_model}' não encontrado. Usando: {models[0]}")

    llama.model = cfg.llama_model
    print(f"  Modelo selecionado: '{cfg.llama_model}' ✓")

    # ── Inicializa Agente ──────────────────────────────────────
    print(f"\n[2/3] Inicializando Agente Central...")
    print(f"  Device  : {cfg.device}")
    print(f"  d_e={cfg.d_e}  d_s={cfg.d_s}  d_h={cfg.d_h}  K={cfg.K}")
    agent = CentralAgent(cfg, llama)

    # ── Carrega estado anterior ────────────────────────────────
    print(f"\n[3/3] Carregando estado salvo...")
    agent.load()

    print(f"\n{sep}")
    print(f"  Sistema pronto!  Modelo: {cfg.llama_model}")
    print(f"  Memória: {len(agent.memory)} experiências  |  Steps: {agent.step_count}")
    print(f"{sep}\n")

    return llama, agent


def main():
    cfg = Config()
    _, agent = setup(cfg)
    _install_exit_handlers(agent)

    from chat import interactive_chat
    try:
        interactive_chat(agent, cfg)
    finally:
        _safe_save(agent, "saída do chat")
        print("Até logo!")


if __name__ == "__main__":
    main()
