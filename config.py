"""
参考仓库 / 文档：
- GitHub: KVignesh122/AssetNewsSentimentAnalyzer
  用途：借鉴“资产新闻分析工具”的工作流组织方式
- GitHub: alphavantage/alpha_vantage_mcp
  用途：借鉴黄金与宏观数据来源设计
- DeepSeek API Docs
  用途：DeepSeek 使用 OpenAI 兼容接口，base_url 指向 https://api.deepseek.com
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    alpha_vantage_api_key: str
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    # 你也可以以后切换成 deepseek-reasoner。
    # 但 DeepSeek 官方文档说明 deepseek-reasoner 的部分采样参数不会生效，
    # 当前 MVP 先用 deepseek-chat 更稳妥。

    news_max_items: int = 8
    article_timeout_sec: int = 10
    request_timeout_sec: int = 15

    @staticmethod
    def from_env() -> "Settings":
        alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()

        missing = []
        if not alpha_vantage_api_key:
            missing.append("ALPHA_VANTAGE_API_KEY")
        if not deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")

        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                f"缺少环境变量：{joined}。"
                "请先在终端中导出这些变量，再启动应用。"
            )

        return Settings(
            alpha_vantage_api_key=alpha_vantage_api_key,
            deepseek_api_key=deepseek_api_key,
        )