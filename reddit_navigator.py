"""Acoes explicitas de interface para Reddit no navegador."""

from __future__ import annotations

import time

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import pyperclip
except Exception:
    pyperclip = None


class RedditNavigator:
    """Executa acoes somente quando o usuario mandar explicitamente."""

    def __init__(self):
        if pyautogui is None:
            raise RuntimeError("pyautogui nao instalado. Instale as dependencias do assistente Reddit.")
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.2

    def focus_post(self, region):
        x, y, w, _h = region
        pyautogui.click(x + min(w // 2, 420), y + 28)
        time.sleep(1.0)

    def go_back(self):
        pyautogui.hotkey("alt", "left")
        time.sleep(0.8)

    def upvote_current(self):
        pyautogui.press("u")
        time.sleep(0.3)

    def open_comment_box(self):
        pyautogui.press("c")
        time.sleep(0.5)

    def paste_text(self, text: str):
        if pyperclip is None:
            raise RuntimeError("pyperclip nao instalado. Instale as dependencias do assistente Reddit.")
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.3)

    def submit_comment(self):
        pyautogui.hotkey("ctrl", "enter")
        time.sleep(0.7)