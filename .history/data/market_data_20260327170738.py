"""
参考仓库 / 文档：
- GitHub: alphavantage/alpha_vantage_mcp
  用途：参考黄金/宏观数据来源设计
- 当前文件的实现策略：
  1. 价格数据使用 Alpha Vantage 免费可用的 TIME_SERIES_DAILY
  2. SMA / RSI 不再调用远程技术指标接口，改为本地计算，减少请求数
  3. 为避免代理/TLS问题，requests.Session() 设置 trust_env=False
  4. 为适配免费 key 的速率限制，加入最小间隔与简单退避重试
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


@dataclass
class MarketSnapshot:
    price_df: pd.DataFrame
    sma_20_df: pd.DataFrame
    rsi_14_df: pd.DataFrame
    treasury_10y_df: pd.DataFrame
    cpi_df: pd.DataFrame


class AlphaVantageClient:
    def __init__(self, api_key: str, timeout: int = 15) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self._last_request_ts: float = 0.0

        self.session = requests.Session()
        self.session.trust_env = False  # 不读取系统/终端代理环境变量

        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update(
            {
                "User-Agent": "gold-insight/0.1 (+python requests)"
            }
        )

    def _wait_for_rate_limit(self, minimum_interval_sec: float = 1.2) -> None:
        """
        Alpha Vantage 免费 key 有 1 request/second 的速率限制。
        这里做一个简单串行节流。
        """
        now = time.time()
        elapsed = now - self._last_request_ts
        if elapsed < minimum_interval_sec:
            time.sleep(minimum_interval_sec - elapsed)

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        统一 GET 封装：
        - 自动带 apikey
        - 简单节流
        - 对 Information 字段做退避重试
        - 避免环境代理导致的 TLS / proxy 问题
        """
        full_params = {**params, "apikey": self.api_key}
        last_info: Optional[str] = None

        for attempt in range(4):
            if attempt == 0:
                self._wait_for_rate_limit(1.2)
            else:
                # 被限流或返回 Information 后做退避
                time.sleep(2.0 * attempt)

            try:
                resp = self.session.get(
                    ALPHA_VANTAGE_BASE_URL,
                    params=full_params,
                    timeout=self.timeout,
                )
                self._last_request_ts = time.time()
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(
                    "请求 Alpha Vantage 失败。"
                    f"\n请求参数: {full_params}"
                    f"\n原始异常: {repr(exc)}"
                ) from exc

            try:
                data = resp.json()
            except Exception as exc:
                raise RuntimeError(
                    "Alpha Vantage 返回了非 JSON 内容。"
                    f"\nHTTP 状态码: {resp.status_code}"
                    f"\n响应前 500 字符: {resp.text[:500]}"
                ) from exc

            if "Error Message" in data:
                raise RuntimeError(f"Alpha Vantage 返回错误：{data['Error Message']}")

            if "Note" in data:
                raise RuntimeError(f"Alpha Vantage 请求被限流：{data['Note']}")

            if "Information" in data:
                last_info = data["Information"]
                # 某些情况下 Information 是瞬时限流提示，尝试重试几次
                continue

            return data

        raise RuntimeError(f"Alpha Vantage 返回提示：{last_info}")

    def get_daily(self, symbol: str = "GLD", outputsize: str = "compact") -> pd.DataFrame:
        """
        获取日线价格数据。
        注意：
        - 使用 TIME_SERIES_DAILY，而不是 premium 的 ADJUSTED / full 组合
        - 这里默认 symbol=GLD，把黄金 ETF 作为黄金市场代理
        """
        data = self._get(
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": outputsize,
            }
        )

        key = "Time Series (Daily)"
        ts = data.get(key, {})
        if not ts:
            raise RuntimeError(f"未获取到 {symbol} 的日线数据。原始返回：{data}")

        rows: list[dict[str, Any]] = []
        for date_str, payload in ts.items():
            rows.append(
                {
                    "date": pd.to_datetime(date_str),
                    "open": float(payload["1. open"]),
                    "high": float(payload["2. high"]),
                    "low": float(payload["3. low"]),
                    "close": float(payload["4. close"]),
                    # 为兼容旧逻辑保留 adjusted_close 字段
                    "adjusted_close": float(payload["4. close"]),
                    "volume": float(payload["5. volume"]),
                }
            )

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def get_treasury_10y(self) -> pd.DataFrame:
        """
        获取美国 10 年期国债收益率。
        """
        data = self._get(
            {
                "function": "TREASURY_YIELD",
                "interval": "daily",
                "maturity": "10year",
            }
        )
        items = data.get("data", [])
        if not items:
            raise RuntimeError("未获取到 10Y Treasury Yield 数据。")

        rows = [
            {"date": pd.to_datetime(item["date"]), "value": float(item["value"])}
            for item in items
            if item.get("value") not in (None, ".", "")
        ]
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def get_cpi(self) -> pd.DataFrame:
        """
        获取美国 CPI（月度）。
        """
        data = self._get({"function": "CPI", "interval": "monthly"})
        items = data.get("data", [])
        if not items:
            raise RuntimeError("未获取到 CPI 数据。")

        rows = [
            {"date": pd.to_datetime(item["date"]), "value": float(item["value"])}
            for item in items
            if item.get("value") not in (None, ".", "")
        ]
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    @staticmethod
    def compute_sma(price_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        本地计算 SMA，避免额外远程请求。
        """
        if price_df.empty:
            return pd.DataFrame(columns=["date", "sma"])

        df = price_df[["date", "close"]].copy()
        df["sma"] = df["close"].rolling(window=window, min_periods=window).mean()
        return df[["date", "sma"]].dropna().reset_index(drop=True)

    @staticmethod
    def compute_rsi(price_df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        本地计算 RSI（Wilder 风格的 EMA 平滑近似）。
        """
        if price_df.empty:
            return pd.DataFrame(columns=["date", "rsi"])

        df = price_df[["date", "close"]].copy()
        delta = df["close"].diff()

        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        # 避免除零
        avg_loss = avg_loss.replace(0, pd.NA)
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        return df[["date", "rsi"]].dropna().reset_index(drop=True)

    def get_market_snapshot(self, symbol: str = "GLD") -> MarketSnapshot:
        """
        保留这个高层接口，便于其他代码复用。
        但 app.py 中会采用拆分缓存方案，不建议频繁直接调用此方法。
        """
        price_df = self.get_daily(symbol=symbol, outputsize="compact")
        sma_20_df = self.compute_sma(price_df, window=20)
        rsi_14_df = self.compute_rsi(price_df, period=14)
        treasury_10y_df = self.get_treasury_10y()
        cpi_df = self.get_cpi()

        return MarketSnapshot(
            price_df=price_df,
            sma_20_df=sma_20_df,
            rsi_14_df=rsi_14_df,
            treasury_10y_df=treasury_10y_df,
            cpi_df=cpi_df,
        )