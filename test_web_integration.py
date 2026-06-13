#!/usr/bin/env python3
"""
Testa se a integração de web_automation com agent e chat está funcionando.
"""
import sys
sys.path.insert(0, ".")

from config import Config
from brain.agent import CentralAgent

def test_import():
    """Verifica se todos os módulos importam corretamente."""
    print("✓ Testing imports...")
    try:
        from web_automation import interact_with_web_ai, WebAIInteractor
        print("  ✓ web_automation imports OK")
    except ImportError as e:
        print(f"  ✗ web_automation import failed: {e}")
        return False
    
    try:
        from web_search import get_searcher
        print("  ✓ web_search imports OK")
    except ImportError as e:
        print(f"  ✗ web_search import failed: {e}")
        return False
    
    return True

def test_agent_method():
    """Testa se o método chat_with_external_ai existe no agente."""
    print("\n✓ Testing agent method...")
    from llama_bridge import LlamaBridge
    
    cfg = Config()
    llama = LlamaBridge(cfg)
    agent = CentralAgent(cfg, llama)
    
    if not hasattr(agent, "chat_with_external_ai"):
        print("  ✗ Agent doesn't have chat_with_external_ai method")
        return False
    
    print("  ✓ Agent has chat_with_external_ai method")
    
    # Teste sem conectar (apenas validação de syntax)
    result = agent.chat_with_external_ai(
        url="https://test.invalid",
        user_message="test",
        store=False
    )
    print(f"  ✓ Method callable (result type: {type(result).__name__})")
    return True

def test_requirements():
    """Verifica se todos os requirements estão instalados."""
    print("\n✓ Testing requirements...")
    requirements = {
        "torch": "torch",
        "requests": "requests",
        "numpy": "numpy",
        "colorama": "colorama",
        "beautifulsoup4": "bs4",
        "duckduckgo_search": "duckduckgo_search",
        "playwright": "playwright",
    }
    
    for pkg_name, import_name in requirements.items():
        try:
            __import__(import_name)
            print(f"  ✓ {pkg_name} installed")
        except ImportError:
            print(f"  ✗ {pkg_name} NOT installed")
            return False
    
    return True

def main():
    print("=" * 60)
    print("Web Integration Test Suite")
    print("=" * 60)
    
    all_ok = True
    
    if not test_import():
        all_ok = False
    
    if not test_requirements():
        all_ok = False
    
    if not test_agent_method():
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All tests passed!")
        print("\nYou can now use:")
        print("  /web_chat <url> <message>")
        print("\nExample:")
        print("  /web_chat https://chatgpt.com What is AI?")
    else:
        print("✗ Some tests failed. Check output above.")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
