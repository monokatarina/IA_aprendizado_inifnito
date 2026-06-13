"""
Bot mecanico por imagem para interacao com IAs web.

Fluxo:
1) Localiza caixa de dialogo por template e clica
2) Cola mensagem
3) Localiza botao enviar por template e clica
4) Aguarda resposta
5) Localiza botao copiar por template e clica
6) Le resposta do clipboard
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import pyautogui
    import pyperclip

    HAS_GUI_AUTOMATION = True
except ImportError:
    HAS_GUI_AUTOMATION = False

@dataclass
class ClickBotConfig:
    dialog_img: str = "templates/web_click/dialog.png"
    send_img: str = "templates/web_click/send.png"
    copy_img: str = "templates/web_click/copy.png"
    confidence: float = 0.78
    retries: int = 8
    retry_interval_s: float = 0.6
    response_wait_s: float = 100.0
    pre_click_delay_s: float = 0.2
    capture_attempts: int = 5
    copy_poll_timeout_s: float = 24.0


class ImageClickBot:
    def __init__(self, cfg: Optional[ClickBotConfig] = None):
        self.cfg = cfg or ClickBotConfig()
        self._supports_confidence = True

        if HAS_GUI_AUTOMATION:
            pyautogui.PAUSE = 0.12
            pyautogui.FAILSAFE = True

    def _validate(self) -> Tuple[bool, str]:
        if not HAS_GUI_AUTOMATION:
            return (
                False,
                "Dependencias ausentes. Instale: pyautogui, pyperclip, pillow (e opencv-python para confidence).",
            )

        missing = []
        for p in [self.cfg.dialog_img, self.cfg.send_img]:
            if not os.path.exists(p):
                missing.append(p)

        if missing:
            return (
                False,
                "Templates nao encontrados: " + ", ".join(missing),
            )

        if not os.path.exists(self.cfg.copy_img):
            print(f"  [click_bot] Aviso: template de copiar ausente ({self.cfg.copy_img}).")

        return True, "ok"

    def _locate_center(self, template_path: str) -> Optional[Tuple[int, int]]:
        last_err = None
        for _ in range(max(1, self.cfg.retries)):
            try:
                if self._supports_confidence:
                    point = pyautogui.locateCenterOnScreen(
                        template_path,
                        confidence=self.cfg.confidence,
                        grayscale=True,
                    )
                else:
                    # Fallback sem OpenCV: confidence nao e suportado.
                    point = pyautogui.locateCenterOnScreen(
                        template_path,
                        grayscale=True,
                    )
                if point is not None:
                    return int(point.x), int(point.y)
            except Exception as e:
                msg = str(e).lower()
                if "confidence keyword argument is only available if opencv is installed" in msg:
                    self._supports_confidence = False
                    last_err = "OpenCV ausente: usando fallback sem confidence."
                    continue
                last_err = e
            time.sleep(self.cfg.retry_interval_s)

        if last_err is not None:
            print(f"  [click_bot] Aviso ao buscar template '{template_path}': {last_err}")
        return None

    def _locate_all_centers(self, template_path: str) -> List[Tuple[int, int]]:
        """Localiza todos os matches de um template na tela."""
        last_err = None
        for _ in range(max(1, self.cfg.retries)):
            try:
                if self._supports_confidence:
                    boxes = list(
                        pyautogui.locateAllOnScreen(
                            template_path,
                            confidence=self.cfg.confidence,
                            grayscale=True,
                        )
                    )
                else:
                    boxes = list(
                        pyautogui.locateAllOnScreen(
                            template_path,
                            grayscale=True,
                        )
                    )

                if boxes:
                    centers: List[Tuple[int, int]] = []
                    for b in boxes:
                        cx = int(b.left + b.width / 2)
                        cy = int(b.top + b.height / 2)
                        centers.append((cx, cy))
                    return centers
            except Exception as e:
                msg = str(e).lower()
                if "confidence keyword argument is only available if opencv is installed" in msg:
                    self._supports_confidence = False
                    last_err = "OpenCV ausente: usando fallback sem confidence."
                    continue
                last_err = e

            time.sleep(self.cfg.retry_interval_s)

        if last_err is not None:
            print(f"  [click_bot] Aviso ao buscar todos os templates '{template_path}': {last_err}")
        return []

    def _click_point(self, point: Tuple[int, int], label: str) -> bool:
        x, y = point
        time.sleep(self.cfg.pre_click_delay_s)
        pyautogui.moveTo(x, y, duration=0.12)
        pyautogui.click(x, y)
        print(f"  [click_bot] Clique em {label} em ({x},{y}).")
        return True

    def _click_template(self, template_path: str, label: str) -> bool:
        point = self._locate_center(template_path)
        if point is None:
            print(f"  [click_bot] Nao encontrei o template de {label}: {template_path}")
            return False

        return self._click_point(point, label)

    def _pick_response_copy_button(self, dialog_point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Escolhe o botao de copiar imediatamente acima da caixa de dialogo.
        Heurística:
        - Só considera botões acima da caixa de diálogo (y < y_dialogo).
        - Entre eles, pega o mais próximo (maior y, mas ainda < y_dialogo).
        - Em empate, escolhe o mais alinhado no eixo X com a caixa de diálogo.
        """
        candidates = self._locate_all_centers(self.cfg.copy_img)
        if not candidates:
            return None

        x_dialog, y_dialog = dialog_point
        # Só botões acima da caixa de diálogo
        above = [p for p in candidates if p[1] < y_dialog]
        if not above:
            print("  [click_bot] Nenhum botão de copiar acima da caixa de diálogo.")
            return None
        # Escolhe o mais próximo (maior y, mas ainda < y_dialog)
        best = sorted(above, key=lambda p: (-p[1], abs(p[0] - x_dialog)))[0]
        print(
            f"  [click_bot] {len(candidates)} botoes 'copiar' detectados; "
            f"selecionado o da resposta em ({best[0]},{best[1]})."
        )
        return best

    def _read_clipboard(self) -> str:
        """Le clipboard com pyperclip e fallback PowerShell no Windows."""
        text = ""
        try:
            text = pyperclip.paste() or ""
        except Exception:
            text = ""

        if text.strip():
            return text

        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-Clipboard -Raw",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                return (result.stdout or "").strip()
        except Exception:
            pass

        return ""


    def run_once(self, message: str) -> str:
        ok, why = self._validate()
        if not ok:
            return f"Falha de configuracao: {why}"

        msg = (message or "").strip()
        if not msg:
            return "Mensagem vazia."

        # 1) Caixa de dialogo
        dialog_point = self._locate_center(self.cfg.dialog_img)
        if dialog_point is None:
            print(f"  [click_bot] Nao encontrei o template de caixa de dialogo: {self.cfg.dialog_img}")
            return "Campo de dialogo nao encontrado na tela."
        self._click_point(dialog_point, "caixa de dialogo")

        # 2) Colar mensagem
        try:
            pyperclip.copy(msg)
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
            pyautogui.hotkey("ctrl", "v")
            print("  [click_bot] Mensagem colada.")
        except Exception as e:
            return f"Erro ao colar mensagem: {e}"


        # 3) Botao enviar
        if not self._click_template(self.cfg.send_img, "botao enviar"):
            return "Botao enviar nao encontrado na tela."

        # 4) Delay fixo de 20s antes de procurar o botão de copiar
        print(f"  [click_bot] Esperando 20s antes de procurar o botão de copiar...")
        time.sleep(20)

        # 5) Espera até o botão de copiar aparecer (sem timeout)
        print(f"  [click_bot] Aguardando aparecer o botão de copiar...")
        copy_point = None
        while True:
            copy_point = self._pick_response_copy_button(dialog_point)
            if copy_point is not None:
                break
            time.sleep(0.5)

        # 6) Botao copiar
        marker = f"CLICKBOT_MARKER_{int(time.time() * 1000)}"
        try:
            pyperclip.copy(marker)
        except Exception:
            pass

        before_clip = self._read_clipboard()

        if not os.path.exists(self.cfg.copy_img):
            return "Template de copiar nao existe."

        for attempt in range(1, max(1, self.cfg.capture_attempts) + 1):
            print(f"  [click_bot] Tentativa de captura {attempt}/{self.cfg.capture_attempts}...")

            self._click_point(copy_point, "botao copiar (resposta)")

            # Pequeno delay e novo clique para garantir
            time.sleep(0.35)
            self._click_point(copy_point, "botao copiar (resposta, 2a tentativa)")

            # Captura clipboard com timeout maior
            poll_end = time.time() + max(1.0, self.cfg.copy_poll_timeout_s)
            while time.time() < poll_end:
                time.sleep(0.25)
                now_clip = self._read_clipboard()
                if now_clip and now_clip != before_clip and now_clip != marker:
                    return now_clip.strip()

        return "Nao consegui capturar texto novo do clipboard apos clicar em copiar."


def mechanical_web_chat(
    message: str,
    dialog_img: Optional[str] = None,
    send_img: Optional[str] = None,
    copy_img: Optional[str] = None,
    confidence: float = 0.78,
    response_wait_s: float = 20.0,
    capture_attempts: int = 5,
) -> str:
    cfg = ClickBotConfig(
        dialog_img=dialog_img or "templates/web_click/dialog.png",
        send_img=send_img or "templates/web_click/send.png",
        copy_img=copy_img or "templates/web_click/copy.png",
        confidence=confidence,
        response_wait_s=response_wait_s,
        capture_attempts=capture_attempts,
    )
    bot = ImageClickBot(cfg)
    return bot.run_once(message)
