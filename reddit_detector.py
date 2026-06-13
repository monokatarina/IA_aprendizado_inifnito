"""Detector OCR simples para posts visiveis do Reddit na tela."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import pytesseract
except Exception:
    pytesseract = None

from PIL import ImageOps

from config import Config


@dataclass
class DetectedPost:
    index: int
    region: Tuple[int, int, int, int]
    anchor_text: str
    preview: str


class RedditScreenDetector:
    """Usa OCR para encontrar posts visiveis no feed do Reddit."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if pytesseract is not None and cfg.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

    def _ensure_ready(self):
        if pyautogui is None:
            raise RuntimeError("pyautogui nao instalado. Instale as dependencias do assistente Reddit.")
        if pytesseract is None:
            raise RuntimeError("pytesseract nao instalado. Instale as dependencias do assistente Reddit.")

    def capture_screen(self):
        self._ensure_ready()
        return pyautogui.screenshot()

    def find_posts(self, max_posts: Optional[int] = None) -> List[DetectedPost]:
        self._ensure_ready()
        screenshot = self.capture_screen()
        gray = ImageOps.grayscale(screenshot)
        data = pytesseract.image_to_data(
            gray,
            lang=self.cfg.reddit_ocr_lang,
            config="--oem 3 --psm 6",
            output_type=pytesseract.Output.DICT,
        )

        anchors: List[Tuple[int, int, str]] = []
        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text = (data["text"][i] or "").strip()
            conf_raw = str(data["conf"][i]).strip()
            try:
                conf = float(conf_raw)
            except ValueError:
                conf = -1.0

            if conf < 25:
                continue
            if re.match(r"^(u/|r/)", text, flags=re.IGNORECASE):
                anchors.append((int(data["left"][i]), int(data["top"][i]), text))

        anchors.sort(key=lambda item: item[1])
        deduped: List[Tuple[int, int, str]] = []
        min_gap = max(60, self.cfg.reddit_post_height // 3)
        for anchor in anchors:
            if not deduped or abs(anchor[1] - deduped[-1][1]) > min_gap:
                deduped.append(anchor)

        limit = max_posts or self.cfg.reddit_scan_max_posts
        screen_w, screen_h = screenshot.size
        posts: List[DetectedPost] = []
        for idx, (x, y, text) in enumerate(deduped[:limit], start=1):
            left = max(0, x - 20)
            top = max(0, y - 50)
            width = min(self.cfg.reddit_post_width, screen_w - left)
            height = min(self.cfg.reddit_post_height, screen_h - top)
            region = (left, top, width, height)
            preview = self.extract_text(region, screenshot=screenshot, max_chars=220)
            posts.append(DetectedPost(index=idx, region=region, anchor_text=text, preview=preview))
        return posts

    def extract_text(self, region: Tuple[int, int, int, int], screenshot=None, max_chars: Optional[int] = None) -> str:
        self._ensure_ready()
        if screenshot is None:
            screenshot = self.capture_screen()
        x, y, w, h = region
        crop = screenshot.crop((x, y, x + w, y + h))
        gray = ImageOps.grayscale(crop)
        text = pytesseract.image_to_string(gray, lang=self.cfg.reddit_ocr_lang, config="--oem 3 --psm 6")
        text = re.sub(r"\s+", " ", text).strip()
        if max_chars is None:
            max_chars = self.cfg.reddit_post_text_max_chars
        return text[:max_chars]