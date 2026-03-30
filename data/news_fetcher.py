"""
参考仓库 / 文档：
- GitHub: KVignesh122/AssetNewsSentimentAnalyzer
  用途：借鉴“Google 新闻检索 + 文章正文抽取 + 送入 LLM 分析”的主流程
- 本文件没有照搬上游代码，而是做了一个更轻量的实现：
  1. 用 Google News RSS 拿新闻列表
  2. 用 readability-lxml 抽取正文
  3. 返回给服务层做 LLM 分析
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote

import feedparser
import requests
from bs4 import BeautifulSoup
from readability import Document


@dataclass
class NewsArticle:
    title: str
    link: str
    published: str
    source: str
    summary: str
    content: str


class GoldNewsFetcher:
    def __init__(self, timeout: int = 10) -> None:
        self.timeout = timeout

    def fetch_google_news_rss(
        self,
        query: str,
        max_items: int = 8,
    ) -> list[NewsArticle]:
        rss_url = (
            "https://news.google.com/rss/search"
            f"?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
        )

        feed = feedparser.parse(rss_url)
        articles: list[NewsArticle] = []

        for entry in feed.entries[:max_items]:
            link = entry.get("link", "")
            title = entry.get("title", "")
            published = entry.get("published", "")
            source = ""
            if "source" in entry and entry.source:
                source = entry.source.get("title", "")

            summary = self._safe_text(entry.get("summary", ""))
            content = self.extract_main_text(link)

            articles.append(
                NewsArticle(
                    title=title,
                    link=link,
                    published=published,
                    source=source,
                    summary=summary,
                    content=content,
                )
            )

        return articles

    def extract_main_text(self, url: str) -> str:
        try:
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                    )
                },
            )
            resp.raise_for_status()
            doc = Document(resp.text)
            html = doc.summary()
            soup = BeautifulSoup(html, "lxml")
            text = soup.get_text(separator="\n", strip=True)
            return text[:6000]  # 控制单篇正文长度，避免 prompt 过长
        except Exception:
            return ""

    @staticmethod
    def _safe_text(raw_html: str) -> str:
        soup = BeautifulSoup(raw_html, "lxml")
        return soup.get_text(separator=" ", strip=True)

    def fetch_default_gold_news(self, max_items: int = 8) -> list[NewsArticle]:
        """
        针对黄金主题给出几个更稳妥的查询词。
        """
        queries = [
            "gold price OR spot gold OR XAUUSD",
            "Fed inflation treasury yield gold",
            "gold safe haven geopolitics",
        ]

        all_articles: list[NewsArticle] = []
        seen_links: set[str] = set()

        for query in queries:
            items = self.fetch_google_news_rss(query=query, max_items=max_items)
            for item in items:
                if item.link and item.link not in seen_links:
                    seen_links.add(item.link)
                    all_articles.append(item)

        # 简单截断，避免新闻过多
        return all_articles[:max_items]