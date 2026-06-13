#!/usr/bin/env python3
"""
Verificação de Setup — IA Local
=================================
Testa se todos os componentes estão instalados e funcionando
antes de rodar main.py.

Uso:
    python setup_check.py
"""

import sys
import traceback


def check(name: str, fn):
    try:
        fn()
        print(f"  ✓ {name}\n")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}\n")
        return False


# ── Verificações ──────────────────────────────────────────────────

def _python():
    v = sys.version_info
    assert v >= (3, 8), f"Python 3.8+ necessário (atual: {v.major}.{v.minor})"
    print(f"    Python {v.major}.{v.minor}.{v.micro}")


def _torch():
    import torch
    print(f"    PyTorch {torch.__version__}")
    dev = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"    Device  : {dev}")
    if torch.cuda.is_available():
        print(f"    GPU     : {torch.cuda.get_device_name(0)}")


def _requests():
    import requests
    print(f"    requests {requests.__version__}")


def _numpy():
    import numpy as np
    print(f"    numpy {np.__version__}")


def _colorama():
    import colorama
    print(f"    colorama {colorama.__version__}")


def _config():
    from config import Config
    import torch
    cfg = Config()
    print(f"    Model  : {cfg.llama_model}")
    print(f"    Device : {cfg.device}")
    print(f"    d_e={cfg.d_e}  d_s={cfg.d_s}  d_h={cfg.d_h}  K={cfg.K}")


def _brain():
    import torch
    from config import Config
    from brain.encoder    import WorldEncoder
    from brain.memory     import EpisodicMemory
    from brain.dynamics   import StateDynamics, SelfPredictor
    from brain.world_model import WorldModel
    from brain.critic     import Critic
    from brain.policy     import InternalPolicy, ExternalPolicy
    from brain.rewards    import IntrinsicRewards
    from brain.gate       import LearningGate

    cfg = Config()

    enc  = WorldEncoder(cfg)
    mem  = EpisodicMemory(cfg)
    dyn  = StateDynamics(cfg)
    sp   = SelfPredictor(cfg)
    wm   = WorldModel(cfg)
    crit = Critic(cfg)
    pi_i = InternalPolicy(cfg)
    pi_e = ExternalPolicy(cfg)
    rw   = IntrinsicRewards(cfg)
    gt   = LearningGate(cfg)

    # Forward pass rápido (sem Ollama)
    raw   = torch.randn(cfg.llama_embed_dim)
    Z     = enc(raw.unsqueeze(0)).squeeze(0)
    assert Z.shape == (cfg.d_e,), f"Encoder: shape errado {Z.shape}"

    S = torch.zeros(cfg.d_s)
    R = torch.zeros(cfg.d_e)
    P = torch.zeros(cfg.d_p)

    a_int  = pi_i(S, R)
    S_new  = dyn(S, Z, a_int, R)
    a_ext  = pi_e(S_new, Z, R, P, training=False)
    Z_hat  = wm(Z, a_ext)
    V      = crit(S_new, Z, R)
    g      = gt(0.5, 0.1)

    print(f"    Z={list(Z.shape)}  S={list(S_new.shape)}  a={list(a_ext.shape)}")
    print(f"    V_t={V.item():.4f}   g_t={g.item():.4f}")


def _ollama():
    import requests
    from config import Config
    cfg = Config()
    try:
        resp = requests.get(f"{cfg.ollama_url}/api/tags", timeout=4)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            print(f"    Ollama rodando em {cfg.ollama_url}")
            print(f"    Modelos : {', '.join(models) if models else 'nenhum'}")
            if not any(cfg.llama_model in m for m in models):
                print(f"    AVISO: '{cfg.llama_model}' não encontrado.")
                print(f"    Execute: ollama pull {cfg.llama_model}")
        else:
            raise ConnectionError(f"HTTP {resp.status_code}")
    except Exception as e:
        raise ConnectionError(
            f"Ollama indisponível: {e}\n"
            "    Inicie com: ollama serve\n"
            f"    Baixe: ollama pull {cfg.llama_model}"
        )


# ── Runner ────────────────────────────────────────────────────────

if __name__ == "__main__":
    sep = "=" * 55
    print(f"\n{sep}")
    print("  VERIFICAÇÃO DE SETUP — IA LOCAL")
    print(sep + "\n")

    checks = [
        ("Python 3.8+",   _python),
        ("PyTorch",        _torch),
        ("requests",       _requests),
        ("numpy",          _numpy),
        ("colorama",       _colorama),
        ("config.py",      _config),
        ("Brain (PyTorch)", _brain),
        ("Ollama / Llama",  _ollama),
    ]

    failed = []
    for name, fn in checks:
        print(f"[{name}]")
        ok = check(name, fn)
        if not ok:
            failed.append(name)

    print(sep)
    if failed:
        print(f"  FALHAS: {', '.join(failed)}")
        print("  Corrija os problemas acima antes de continuar.")
    else:
        print("  Todos os checks passaram!")
        print("  Execute: python main.py")
    print(sep + "\n")
