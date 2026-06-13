#!/usr/bin/env python3
"""
Script para debugar e encontrar os seletores corretos em páginas web.
Abre a página, inspeciona o HTML e mostra elementos que poderiam ser input/botão.
"""

import asyncio
from playwright.async_api import async_playwright


async def debug_page_selectors(url: str):
    """Inspeciona página e mostra seletores disponíveis."""
    print(f"\n{'='*70}")
    print(f"Debugando: {url}")
    print(f"{'='*70}\n")
    
    async with await async_playwright().start() as p:
        browser = await p.chromium.launch(headless=False)  # headless=False para ver a página
        page = await browser.new_page()
        
        try:
            # Navega para a página
            print("📍 Navegando para a página...")
            await page.goto(url, wait_until="networkidle", timeout=30000)
            print("✓ Página carregada\n")
            
            # Aguarda estabilização
            await asyncio.sleep(2)
            
            # 1. PROCURA CAMPOS DE INPUT
            print(f"\n{'─'*70}")
            print("🔍 CAMPOS DE INPUT ENCONTRADOS:")
            print(f"{'─'*70}\n")
            
            inputs = await page.query_selector_all("textarea, input[type='text'], [contenteditable='true']")
            
            if not inputs:
                print("❌ Nenhum campo de input encontrado!")
            else:
                print(f"✓ Encontrados {len(inputs)} campos:\n")
                
                for i, elem in enumerate(inputs, 1):
                    try:
                        # Pega atributos
                        tag = await elem.evaluate("e => e.tagName")
                        id_attr = await elem.evaluate("e => e.id || 'N/A'")
                        cls = await elem.evaluate("e => e.className || 'N/A'")
                        placeholder = await elem.evaluate("e => e.placeholder || 'N/A'")
                        is_visible = await elem.is_visible()
                        
                        print(f"  {i}. <{tag}> - {'VISÍVEL' if is_visible else 'OCULTO'}")
                        print(f"     ID: {id_attr}")
                        print(f"     CLASS: {cls}")
                        print(f"     PLACEHOLDER: {placeholder}\n")
                    except Exception as e:
                        print(f"  {i}. Erro ao inspecionar: {e}\n")
            
            # 2. PROCURA BOTÕES
            print(f"\n{'─'*70}")
            print("🔍 BOTÕES ENCONTRADOS:")
            print(f"{'─'*70}\n")
            
            buttons = await page.query_selector_all("button")
            
            if not buttons:
                print("❌ Nenhum botão encontrado!")
            else:
                print(f"✓ Encontrados {len(buttons)} botões (mostrando primeiros 10):\n")
                
                for i, btn in enumerate(buttons[:10], 1):
                    try:
                        text = await btn.evaluate("e => e.textContent.trim()")
                        id_attr = await btn.evaluate("e => e.id || 'N/A'")
                        cls = await btn.evaluate("e => e.className || 'N/A'")
                        is_visible = await btn.is_visible()
                        
                        # Trunca texto longo
                        if len(text) > 40:
                            text = text[:37] + "..."
                        
                        print(f"  {i}. TEXTO: {text}")
                        print(f"     VISÍVEL: {'✓' if is_visible else '✗'}")
                        print(f"     ID: {id_attr}")
                        print(f"     CLASS: {cls}\n")
                    except Exception as e:
                        print(f"  {i}. Erro ao inspecionar: {e}\n")
            
            # 3. PROCURA DIVS EDITÁVEIS
            print(f"\n{'─'*70}")
            print("🔍 DIVS EDITÁVEIS ENCONTRADOS:")
            print(f"{'─'*70}\n")
            
            divs = await page.query_selector_all("[contenteditable='true']")
            
            if not divs:
                print("❌ Nenhuma div editável encontrada!")
            else:
                print(f"✓ Encontradas {len(divs)} divs:\n")
                
                for i, div in enumerate(divs, 1):
                    try:
                        id_attr = await div.evaluate("e => e.id || 'N/A'")
                        cls = await div.evaluate("e => e.className || 'N/A'")
                        is_visible = await div.is_visible()
                        
                        print(f"  {i}. {'VISÍVEL' if is_visible else 'OCULTO'}")
                        print(f"     ID: {id_attr}")
                        print(f"     CLASS: {cls}\n")
                    except Exception as e:
                        print(f"  {i}. Erro ao inspecionar: {e}\n")
            
            # 4. SUGERE SELETORES
            print(f"\n{'─'*70}")
            print("💡 SELETORES SUGERIDOS:")
            print(f"{'─'*70}\n")
            
            suggestions = [
                ("textarea", "Qualquer textarea"),
                ("input[type='text']", "Qualquer input de texto"),
                ("[contenteditable='true']", "Divs editáveis"),
                ("[role='textbox']", "Elementos com role textbox"),
                ("[role='searchbox']", "Elementos com role searchbox"),
            ]
            
            for selector, desc in suggestions:
                try:
                    elem = await page.query_selector(selector)
                    if elem:
                        is_visible = await elem.is_visible()
                        status = "✓ ENCONTRADO" if is_visible else "✗ OCULTO"
                        print(f"  {selector}")
                        print(f"    └─ {desc}: {status}\n")
                except Exception:
                    pass
            
            # 5. PAUSA PARA O USUÁRIO INSPECIONAR MANUALMENTE
            print(f"\n{'─'*70}")
            print("⏸️  PÁGINA ABERTA PARA INSPEÇÃO MANUAL")
            print(f"{'─'*70}\n")
            print("Dicas para inspeção manual (pressione F12 no navegador aberto):")
            print("  1. Clique no campo de input com o mouse")
            print("  2. No DevTools (F12), vá para Console")
            print("  3. Execute: $0.tagName, $0.id, $0.className")
            print("  4. Execute: document.activeElement.tagName")
            print("\nPressione ENTER aqui para fechar o navegador...\n")
            input()
            
        except Exception as e:
            print(f"❌ Erro: {e}")
        finally:
            await browser.close()


async def main():
    url = input("\nDigite a URL para debugar (ex: https://chat.deepseek.com/): ").strip()
    
    if not url:
        print("URL vazia!")
        return
    
    if not url.startswith("http"):
        url = "https://" + url
    
    await debug_page_selectors(url)


if __name__ == "__main__":
    asyncio.run(main())
