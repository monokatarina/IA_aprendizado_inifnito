"""
Busca web gratuita para a Deorita.

Usa DuckDuckGo sem API paga e faz leitura simples de paginas para transformar
resultado bruto em texto reaproveitavel pela memoria episodica.
"""

from __future__ import annotations

import re
import time
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except Exception:
    DDGS_AVAILABLE = False


class WebSearch:
    """Buscador web sem API paga com historico simples."""

    def __init__(self, max_results: int = 5, timeout: int = 12, page_max_chars: int = 2200):
        self.max_results = max_results
        self.timeout = timeout
        self.page_max_chars = page_max_chars
        self.search_history: List[Dict] = []
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            }
        )

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """Executa busca textual e retorna titulo, url e snippet."""
        query = query.strip()
        if not query:
            return []

        limit = max_results or self.max_results
        results = self._search_ddgs(query, limit)
        if not results:
            results = self._search_duckduckgo_html(query, limit)

        self.search_history.append(
            {
                "query": query,
                "timestamp": time.time(),
                "num_results": len(results),
            }
        )
        return results

    def _search_ddgs(self, query: str, limit: int) -> List[Dict[str, str]]:
        if not DDGS_AVAILABLE:
            return []

        results: List[Dict[str, str]] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=limit, region="wt-wt", safesearch="off"):
                    url = item.get("href") or item.get("url") or ""
                    if not url:
                        continue
                    results.append(
                        {
                            "title": (item.get("title") or url).strip(),
                            "url": url.strip(),
                            "snippet": (item.get("body") or "").strip(),
                        }
                    )
                    if len(results) >= limit:
                        break
        except Exception:
            return []
        return results

    def _search_duckduckgo_html(self, query: str, limit: int) -> List[Dict[str, str]]:
        """Fallback via HTML publico do DuckDuckGo."""
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            return []

        results: List[Dict[str, str]] = []
        for block in soup.select(".result"):
            link = block.select_one("a.result__a")
            snippet = block.select_one(".result__snippet")
            if not link:
                continue
            href = (link.get("href") or "").strip()
            title = link.get_text(" ", strip=True)
            body = snippet.get_text(" ", strip=True) if snippet else ""
            if href:
                results.append({"title": title or href, "url": href, "snippet": body})
            if len(results) >= limit:
                break
        return results

    def read_page(self, url: str, max_chars: Optional[int] = None) -> Optional[str]:
        """Baixa e limpa o texto principal de uma pagina."""
        if not url:
            return None

        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
        except Exception:
            return None

        try:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "svg", "img"]):
                tag.decompose()

            container = soup.find("article") or soup.find("main") or soup.body
            if container is None:
                return None

            text = container.get_text(" ", strip=True)
            text = re.sub(r"\s+", " ", text)
            limit = max_chars or self.page_max_chars
            return text[:limit] if text else None
        except Exception:
            return None

    def search_and_read(
        self,
        query: str,
        read_top_n: int = 1,
        max_results: Optional[int] = None,
    ) -> str:
        """Busca e tenta ler as melhores paginas para formar um resumo bruto."""
        results = self.search(query, max_results=max_results)
        if not results:
            return f"Nao encontrei resultados para: {query}"

        lines = [f"Pesquisa: {query}", f"Resultados: {len(results)}"]
        for idx, res in enumerate(results[: max(read_top_n, 0)], start=1):
            lines.append("")
            lines.append(f"[{idx}] {res['title']}")
            lines.append(f"URL: {res['url']}")
            if res["snippet"]:
                lines.append(f"Snippet: {res['snippet'][:320]}")
            page_text = self.read_page(res["url"])
            if page_text:
                lines.append(f"Conteudo: {page_text[: self.page_max_chars]}")
        return "\n".join(lines)

    def get_search_suggestions(self, topic: str) -> List[str]:
        topic = topic.strip()
        if not topic:
            return []
        return [
            f"O que e {topic}?",
            f"Como funciona {topic}",
            f"Ultimas noticias sobre {topic}",
            f"Riscos e beneficios de {topic}",
            f"Especialistas sobre {topic}",
        ]

    def get_stats(self) -> Dict[str, object]:
        last_query = self.search_history[-1]["query"] if self.search_history else ""
        return {
            "enabled": True,
            "searches": len(self.search_history),
            "last_query": last_query,
        }


_searcher: Optional[WebSearch] = None


def get_searcher(max_results: int = 5, timeout: int = 12, page_max_chars: int = 2200) -> WebSearch:
    global _searcher
    if _searcher is None:
        _searcher = WebSearch(
            max_results=max_results,
            timeout=timeout,
            page_max_chars=page_max_chars,
        )
    return _searcher