"""Assistente Reddit em modo seguro: analisa e so age por comando explicito."""

from __future__ import annotations

from typing import Dict, List

from reddit_brain import DeoritaRedditBrain
from reddit_detector import DetectedPost, RedditScreenDetector
from reddit_navigator import RedditNavigator


class DeoritaRedditAssistant:
    def __init__(self, agent, cfg):
        self.agent = agent
        self.cfg = cfg
        self.detector = RedditScreenDetector(cfg)
        self.navigator = RedditNavigator()
        self.brain = DeoritaRedditBrain(agent)
        self.last_posts: List[DetectedPost] = []
        self.comment_drafts: Dict[int, str] = {}

    def scan(self) -> List[DetectedPost]:
        self.last_posts = self.detector.find_posts(max_posts=self.cfg.reddit_scan_max_posts)
        return self.last_posts

    def inspect(self, index: int) -> Dict[str, object]:
        post = self._get_post(index)
        full_text = self.detector.extract_text(post.region)
        analysis = self.brain.analyze_post(full_text)
        return {
            "index": post.index,
            "preview": post.preview,
            "text": full_text,
            "analysis": analysis,
        }

    def draft(self, index: int) -> Dict[str, object]:
        post = self._get_post(index)
        full_text = self.detector.extract_text(post.region)
        analysis = self.brain.analyze_post(full_text)
        draft = self.brain.draft_comment(full_text)
        self.comment_drafts[index] = draft
        return {
            "index": post.index,
            "preview": post.preview,
            "analysis": analysis,
            "draft": draft,
        }

    def open_post(self, index: int):
        post = self._get_post(index)
        self.navigator.focus_post(post.region)

    def like(self, index: int) -> str:
        post = self._get_post(index)
        self.navigator.focus_post(post.region)
        self.navigator.upvote_current()
        self.navigator.go_back()
        return f"Upvote enviado para o post {post.index}."

    def comment(self, index: int, text: str | None = None) -> str:
        post = self._get_post(index)
        final_text = (text or self.comment_drafts.get(index) or "").strip()
        if len(final_text) < 4:
            raise RuntimeError("Nenhum rascunho disponivel. Use /reddit draft <n> primeiro.")

        self.navigator.focus_post(post.region)
        self.navigator.open_comment_box()
        self.navigator.paste_text(final_text)
        self.navigator.submit_comment()
        self.navigator.go_back()

        self.agent.memory.add_pinned_text(
            Z=self.agent._perceive(final_text),
            text=f"[REDDIT_COMMENT] {final_text[:180]}",
        )
        return f"Comentario enviado no post {post.index}."

    def status(self) -> Dict[str, object]:
        return {
            "posts_scanned": len(self.last_posts),
            "drafts": len(self.comment_drafts),
        }

    def _get_post(self, index: int) -> DetectedPost:
        if not self.last_posts:
            self.scan()
        if not self.last_posts:
            raise RuntimeError("Nenhum post detectado. Deixe o feed do Reddit visivel e use /reddit scan.")
        if index < 1 or index > len(self.last_posts):
            raise RuntimeError(f"Indice invalido: {index}. Ha {len(self.last_posts)} posts detectados.")
        return self.last_posts[index - 1]