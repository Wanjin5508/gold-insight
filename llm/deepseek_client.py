"""
参考仓库 / 文档：
- DeepSeek API Docs
  用途：官方说明可使用 OpenAI SDK，并将 base_url 设为 https://api.deepseek.com
- GitHub: KVignesh122/AssetNewsSentimentAnalyzer
  用途：借鉴“将新闻摘要/正文交给 LLM 进行情绪与报告分析”的思路

说明：
- 当前使用 deepseek-chat。
- 若以后切到 deepseek-reasoner，需要注意官方文档中提到：
  部分采样参数如 temperature 对 reasoner 不生效。
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def analyze_gold_market(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            stream=False,
        )

        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "DeepSeek 返回结果不是合法 JSON。"
                f"原始内容如下：{content}"
            ) from exc