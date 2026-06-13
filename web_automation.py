"""
Automatizador de Interação Web para a Deorita.

Permite que a IA local converse com outras IAs na web:
  - Abre página da IA externa
  - Digita e envia mensagem no campo de input
  - Aguarda resposta
  - Copia o resultado
  - Retorna para a IA local
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Tuple

try:
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class WebAIInteractor:
    """Automatiza interação com IAs hospedadas em páginas web."""

    def __init__(self, timeout_seconds: int = 30, headless: bool = False):
        self.timeout = timeout_seconds * 1000  # Playwright usa ms
        self.headless = headless
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None

    async def _start_browser_async(self):
        """Inicia o navegador Playwright (async)."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright não instalado. Execute: pip install playwright && playwright install"
            )

        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        self.page.set_default_timeout(self.timeout)

    async def _close_browser_async(self):
        """Fecha o navegador (async)."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def navigate_to(self, url: str) -> bool:
        """Navega para a URL."""
        if self.page is None:
            return False
        try:
            await self.page.goto(url, wait_until="networkidle")
            return True
        except Exception as e:
            print(f"  [web_automation] Erro ao navegar para {url}: {e}")
            return False

    async def _find_input_field(self) -> Optional[object]:
        """Tenta encontrar campo de input para chat."""
        if self.page is None:
            return None

        # Estratégias comuns para encontrar campo de input
        # Ordem: mais específico → mais genérico
        selectors = [
            # DeepSeek, ChatGPT, Claude
            'textarea[placeholder*="escrever" i]',
            'textarea[placeholder*="chat" i]',
            'textarea[placeholder*="message" i]',
            'textarea[placeholder*="type" i]',
            'textarea[placeholder*="ask" i]',
            'textarea[placeholder*="input" i]',
            
            # Input fields
            'input[type="text"][placeholder*="escrever" i]',
            'input[type="text"][placeholder*="message" i]',
            'input[type="text"][placeholder*="chat" i]',
            'input[type="text"][placeholder*="type" i]',
            
            # Divs editáveis
            '[contenteditable="true"]',
            
            # Classes comuns
            '[class*="input-box"]',
            '[class*="input-field"]',
            '[class*="message-input"]',
            '[class*="chat-input"]',
            
            # Fallback: qualquer textarea
            'textarea',
            
            # Fallback: qualquer input text
            'input[type="text"]',
        ]

        for selector in selectors:
            try:
                elem = await self.page.query_selector(selector)
                if elem:
                    # Verifica se está visível
                    is_visible = await elem.is_visible()
                    if is_visible:
                        return elem
            except Exception:
                pass

        return None

    async def _find_send_button(self) -> Optional[object]:
        """Tenta encontrar botão de envio."""
        if self.page is None:
            return None

        selectors = [
            # Português
            'button:has-text("Enviar")',
            'button:has-text("enviar")',
            
            # Inglês
            'button:has-text("Send")',
            'button:has-text("send")',
            'button:has-text("Submit")',
            
            # Aria labels
            'button[aria-label*="nv" i]',  # envelope
            'button[aria-label*="send" i]',
            'button[aria-label*="submit" i]',
            
            # Titles
            'button[title*="nv" i]',
            'button[title*="send" i]',
            
            # Classes comuns
            'button[class*="send"]',
            'button[class*="submit"]',
            '[class*="send-button"]',
            '[class*="submit-button"]',
            
            # SVG + button (ícone ao lado)
            'button svg + button',
            'button:has(svg)',
            
            # Tipo submit
            'button[type="submit"]',
            
            # Último botão perto de input
            'button',  # fallback genérico
        ]

        for selector in selectors:
            try:
                btn = await self.page.query_selector(selector)
                if btn:
                    is_visible = await btn.is_visible()
                    if is_visible:
                        return btn
            except Exception:
                pass

        return None

    async def _find_response_text(self) -> Optional[str]:
        """Tenta extrair a resposta mais recente da página."""
        if self.page is None:
            return None

        # Estratégias para encontrar resposta
        strategies = [
            # Procura última mensagem da "assistant" / "bot"
            ('div[data-testid*="message"][data-testid*="assistant"]', "textContent"),
            ('div[class*="assistant"][class*="message"]', "textContent"),
            ('div[role="article"]', "textContent"),  # último article
            ('[class*="bot-message"]', "textContent"),
            ('[class*="ai-message"]', "textContent"),
            ('[class*="assistant-message"]', "textContent"),
            
            # Parágrafos e divs genéricos
            ('p', "textContent"),
            ('div', "textContent"),
        ]

        try:
            # Tenta pegar último elemento de cada estratégia
            for selector, attr in strategies:
                try:
                    elems = await self.page.query_selector_all(selector)
                    if elems:
                        # Pega vários últimos elementos e tenta encontrar uma resposta substantiva
                        for elem in reversed(elems[-5:]):  # últimos 5
                            try:
                                text = await elem.evaluate(f"e => e.{attr}")
                                if isinstance(text, str):
                                    text = text.strip()
                                else:
                                    text = str(text).strip()
                                
                                # Remove espaços em branco excessivos
                                text = " ".join(text.split())
                                
                                # Resposta deve ter pelo menos 20 chars e parecer real
                                if text and len(text) > 20 and not text.startswith("["):
                                    return text
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

        return None

    async def send_message_and_get_response(
        self, user_message: str, wait_seconds: int = 5
    ) -> Optional[str]:
        """
        Escreve mensagem, envia e espera resposta.

        Returns:
            Texto da resposta ou None se falhar.
        """
        if self.page is None:
            return None

        # 1. Encontra e preenche campo de input
        input_field = await self._find_input_field()
        if not input_field:
            print("  [web_automation] Campo de input não encontrado.")
            print("  [web_automation] Tente outra URL ou aguarde o carregamento completo.")
            return None

        try:
            await input_field.scroll_into_view_if_needed()
            await asyncio.sleep(0.5)  # Aguarda scroll render
            await input_field.click()
            await asyncio.sleep(0.2)
            await input_field.fill("")  # limpa campo
            await asyncio.sleep(0.1)
            await input_field.type(user_message, delay=50)  # tipo lento para parecer natural
            print(f"  [web_automation] Mensagem digitada com sucesso.")
        except Exception as e:
            print(f"  [web_automation] Erro ao preencher input: {e}")
            return None

        # 2. Encontra e clica botão de envio
        await asyncio.sleep(0.3)  # Aguarda validação do campo
        send_button = await self._find_send_button()
        if not send_button:
            print("  [web_automation] Botão de envio não encontrado.")
            return None

        try:
            await send_button.click()
            print(f"  [web_automation] Mensagem enviada.")
            await asyncio.sleep(1)  # Aguarda início da resposta
        except Exception as e:
            print(f"  [web_automation] Erro ao clicar envio: {e}")
            return None

        # 3. Aguarda resposta (polling)
        print(f"  [web_automation] Aguardando resposta...")
        start = time.time()
        while time.time() - start < wait_seconds:
            response = await self._find_response_text()
            if response and len(response) > 50:  # resposta deve ser substantiva
                print(f"  [web_automation] Resposta obtida!")
                return response.strip()
            await asyncio.sleep(0.5)

        print(f"  [web_automation] Timeout aguardando resposta ({wait_seconds}s).")
        return None

    async def chat_with_external_ai(self, url: str, user_message: str) -> Optional[str]:
        """Fluxo completo: navega, envia, recebe, fecha."""
        await self._start_browser_async()
        try:
            if not await self.navigate_to(url):
                return None

            # Aguarda a página carregar completamente
            try:
                await self.page.wait_for_load_state("networkidle", timeout=self.timeout)
            except Exception:
                print("  [web_automation] Página ainda carregando, prosseguindo mesmo assim...")
                pass
            
            await asyncio.sleep(2)  # pausa extra para JS renderizar e página estabilizar

            response = await self.send_message_and_get_response(user_message, wait_seconds=10)
            return response
        except Exception as e:
            print(f"  [web_automation] Erro geral: {e}")
            return None
        finally:
            await self._close_browser_async()


async def interact_with_web_ai_async(
    url: str, message: str, headless: bool = True, timeout: int = 30
) -> Optional[str]:
    """
    Função async para conversa única com IA na web.

    Args:
        url: URL da página com a IA
        message: Mensagem a enviar
        headless: Se True, navegador roda invisível
        timeout: Tempo máximo em segundos

    Returns:
        Resposta da IA externa ou None
    """
    interactor = WebAIInteractor(timeout_seconds=timeout, headless=headless)
    return await interactor.chat_with_external_ai(url, message)


def interact_with_web_ai(
    url: str, message: str, headless: bool = True, timeout: int = 30
) -> Optional[str]:
    """
    Função de conveniência (sync wrapper) para conversa com IA na web.
    Executa a operação async em uma nova event loop para compatibilidade.

    Args:
        url: URL da página com a IA
        message: Mensagem a enviar
        headless: Se True, navegador roda invisível
        timeout: Tempo máximo em segundos

    Returns:
        Resposta da IA externa ou None
    """
    try:
        # Tenta usar o event loop existente (se estiver em asyncio context)
        loop = asyncio.get_running_loop()
        # Se chegou aqui, já está em um loop. Isso não deveria acontecer mais.
        raise RuntimeError("Cannot call sync wrapper from inside async context. Use interact_with_web_ai_async instead.")
    except RuntimeError:
        # Não há event loop rodando, podemos criar um novo
        return asyncio.run(interact_with_web_ai_async(url, message, headless, timeout))
