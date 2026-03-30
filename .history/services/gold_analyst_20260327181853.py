"""
参考仓库 / 文档：
- GitHub: KVignesh122/AssetNewsSentimentAnalyzer
  用途：借鉴“新闻 + LLM -> 市场分析结论 / 日报”的工作流
- GitHub: alphavantage/alpha_vantage_mcp
  用途：借鉴黄金与宏观指标的数据范围
- DeepSeek API Docs
  用途：使用 JSON 输出，便于前端稳定解析

本文件职责：
1. 组装市场数据 + 新闻数据
2. 写 prompt
3. 调 DeepSeek 得到结构化分析结论
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from data.market_data import MarketSnapshot
from data.news_fetcher import NewsArticle
from llm.deepseek_client import DeepSeekClient


@dataclass
class AnalysisInput:
    symbol: str
    market: MarketSnapshot
    news: list[NewsArticle]


class GoldAnalystService:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self.llm_client = llm_client

    def run(self, analysis_input: AnalysisInput) -> dict[str, Any]:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(analysis_input)
        return self.llm_client.analyze_gold_market(system_prompt, user_prompt)

    def _build_system_prompt(self) -> str:
        return """
你是一个谨慎、保守、注重证据链的黄金市场分析助手。
你不是交易喊单工具，也不要给出确定性承诺。

请严格输出 JSON，格式如下：
{
  "overall_bias": "bullish|bearish|sideways",
  "one_sentence_summary": "一句话结论",
  "price_signal": {
    "latest_price": "",
    "trend_vs_sma20": "",
    "rsi_status": ""
  },
  "macro_signal": {
    "treasury_yield_signal": "",
    "cpi_signal": ""
  },
  "news_signal": {
    "overall_news_sentiment": "positive|negative|mixed",
    "key_drivers": ["", "", ""]
  },
  "risk_points": ["", "", ""],
  "action_hint": "请给出详细的投资建议和你作为一名资深投资专家对给定信息的分析过程，此外你必须给出你对你的投资决策的信心程度，从 0 分到 100 分"
}

要求：
1. 必须基于给定数据，不要编造缺失事实。
2. 若证据冲突，应明确指出“不确定性”。
3. 新闻部分要综合标题、摘要和正文片段。
4. 输出必须是合法 JSON。
5. 
""".strip()

    def _build_user_prompt(self, analysis_input: AnalysisInput) -> str:
        market = analysis_input.market

        latest_price_row = market.price_df.iloc[-1]
        latest_sma_row = market.sma_20_df.iloc[-1]
        latest_rsi_row = market.rsi_14_df.iloc[-1]
        latest_treasury_row = market.treasury_10y_df.iloc[-1]
        latest_cpi_row = market.cpi_df.iloc[-1]

        last_20 = market.price_df.tail(20)[["date", "close"]].copy()
        last_20["date"] = last_20["date"].dt.strftime("%Y-%m-%d")

        news_blocks: list[str] = []
        for idx, article in enumerate(analysis_input.news, start=1):
            content = article.content[:1200] if article.content else ""
            news_blocks.append(
                f"""
[新闻 {idx}]
标题: {article.title}
来源: {article.source}
发布时间: {article.published}
链接: {article.link}
摘要: {article.summary}
正文片段: {content}
""".strip()
            )

        last_20_text = "\n".join(
            f"{row['date']}: {row['close']:.2f}" for _, row in last_20.iterrows()
        )

        user_prompt = f"""
分析标的: {analysis_input.symbol}

【价格与技术指标】
最新收盘价: {latest_price_row["close"]:.2f}
最新调整收盘价: {latest_price_row["adjusted_close"]:.2f}
最新 SMA20: {latest_sma_row["sma"]:.2f}
最新 RSI14: {latest_rsi_row["rsi"]:.2f}

最近 20 个交易日收盘价:
{last_20_text}

【宏观指标】
最新美国10年期国债收益率:
日期: {latest_treasury_row["date"].strftime("%Y-%m-%d")}
值: {latest_treasury_row["value"]:.4f}

最新美国 CPI:
日期: {latest_cpi_row["date"].strftime("%Y-%m-%d")}
值: {latest_cpi_row["value"]:.4f}

【相关新闻】
{chr(10).join(news_blocks)}

请根据以上输入，判断黄金市场短期偏多、偏空还是震荡，
并解释主要依据、风险点以及一个谨慎的观察建议。
""".strip()

        return user_prompt