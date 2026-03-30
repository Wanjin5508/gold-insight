"""
参考仓库 / 文档：
- GitHub: alphavantage/alpha_vantage_mcp
  用途：该仓库 README 列出了 GOLD_SILVER_SPOT / GOLD_SILVER_HISTORY 等贵金属数据能力。
  这里不直接接 MCP，而是走更简单的 Alpha Vantage REST 接口，适合本地 Python 工具。
- 本文件职责：
  1. 拉取黄金日线价格（这里以 GLD ETF 作为一个稳定、易获取的黄金代理）
  2. 拉取技术指标（SMA / RSI）
  3. 拉取宏观指标（10Y Treasury Yield / CPI）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests


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

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        full_params = {**params, "apikey": self.api_key}
        resp = requests.get(ALPHA_VANTAGE_BASE_URL, params=full_params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        # Alpha Vantage 常见限流/异常返回
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage 返回错误：{data['Error Message']}")
        if "Note" in data:
            raise RuntimeError(
                "Alpha Vantage 请求被限流或达到免费额度限制。"
                f"返回内容：{data['Note']}"
            )
        return data

    def get_daily_adjusted(self, symbol: str = "GLD", outputsize: str = "compact") -> pd.DataFrame:
        """
        获取日线数据。这里用 GLD 作为黄金市场代理。
        若你后续更想贴近期货/现货，可以再替换成别的数据源。
        """
        data = self._get(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": outputsize,
            }
        )
        key = "Time Series (Daily)"
        ts = data.get(key, {})
        if not ts:
            raise RuntimeError("未获取到黄金代理日线数据。")

        rows: list[dict[str, Any]] = []
        for date_str, payload in ts.items():
            rows.append(
                {
                    "date": pd.to_datetime(date_str),
                    "open": float(payload["1. open"]),
                    "high": float(payload["2. high"]),
                    "low": float(payload["3. low"]),
                    "close": float(payload["4. close"]),
                    "adjusted_close": float(payload["5. adjusted close"]),
                    "volume": float(payload["6. volume"]),
                }
            )
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        return df

    def get_sma(
        self,
        symbol: str = "GLD",
        interval: str = "daily",
        time_period: int = 20,
        series_type: str = "close",
    ) -> pd.DataFrame:
        data = self._get(
            {
                "function": "SMA",
                "symbol": symbol,
                "interval": interval,
                "time_period": time_period,
                "series_type": series_type,
            }
        )
        key = "Technical Analysis: SMA"
        ts = data.get(key, {})
        if not ts:
            raise RuntimeError("未获取到 SMA 数据。")

        rows = [
            {"date": pd.to_datetime(date_str), "sma": float(payload["SMA"])}
            for date_str, payload in ts.items()
        ]
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def get_rsi(
        self,
        symbol: str = "GLD",
        interval: str = "daily",
        time_period: int = 14,
        series_type: str = "close",
    ) -> pd.DataFrame:
        data = self._get(
            {
                "function": "RSI",
                "symbol": symbol,
                "interval": interval,
                "time_period": time_period,
                "series_type": series_type,
            }
        )
        key = "Technical Analysis: RSI"
        ts = data.get(key, {})
        if not ts:
            raise RuntimeError("未获取到 RSI 数据。")

        rows = [
            {"date": pd.to_datetime(date_str), "rsi": float(payload["RSI"])}
            for date_str, payload in ts.items()
        ]
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def get_treasury_10y(self) -> pd.DataFrame:
        """
        拉取美国 10 年期国债收益率。
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
        拉取美国 CPI。
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

    def get_market_snapshot(self, symbol: str = "GLD") -> MarketSnapshot:
        price_df = self.get_daily_adjusted(symbol=symbol)
        sma_20_df = self.get_sma(symbol=symbol, time_period=20)
        rsi_14_df = self.get_rsi(symbol=symbol, time_period=14)
        treasury_10y_df = self.get_treasury_10y()
        cpi_df = self.get_cpi()

        return MarketSnapshot(
            price_df=price_df,
            sma_20_df=sma_20_df,
            rsi_14_df=rsi_14_df,
            treasury_10y_df=treasury_10y_df,
            cpi_df=cpi_df,
        )