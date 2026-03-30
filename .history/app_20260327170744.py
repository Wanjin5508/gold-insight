"""
参考仓库 / 文档：
- GitHub: KVignesh122/AssetNewsSentimentAnalyzer
  用途：参考“新闻驱动的资产分析工具”整体定位
- GitHub: alphavantage/alpha_vantage_mcp
  用途：参考黄金/宏观数据层设计
- DeepSeek API Docs
  用途：DeepSeek OpenAI-compatible 接法

当前修正版重点：
1. 将市场数据缓存拆分，降低 Alpha Vantage 免费额度消耗
2. 价格/技术指标、收益率、CPI 分别设置不同 TTL
3. 保留新闻抓取 + DeepSeek 分析流程
"""

from __future__ import annotations

import traceback

import pandas as pd
import streamlit as st

from config import Settings
from data.market_data import AlphaVantageClient, MarketSnapshot
from data.news_fetcher import GoldNewsFetcher
from llm.deepseek_client import DeepSeekClient
from services.gold_analyst import AnalysisInput, GoldAnalystService


st.set_page_config(page_title="Gold Insight (DeepSeek)", layout="wide")
st.title("Gold Insight · DeepSeek 版")
st.caption("黄金市场分析工具：价格 + 宏观 + 新闻 + DeepSeek 结构化结论")


@st.cache_resource
def load_settings() -> Settings:
    return Settings.from_env()


@st.cache_resource
def build_clients(settings: Settings):
    market_client = AlphaVantageClient(
        settings.alpha_vantage_api_key,
        timeout=settings.request_timeout_sec,
    )
    news_fetcher = GoldNewsFetcher(timeout=settings.article_timeout_sec)
    llm_client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
    )
    analyst_service = GoldAnalystService(llm_client)
    return market_client, news_fetcher, analyst_service


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_price_snapshot(symbol: str):
    """
    价格与技术指标缓存 1 小时：
    - 日线价格
    - 本地计算的 SMA20 / RSI14
    """
    settings = load_settings()
    market_client, _, _ = build_clients(settings)

    price_df = market_client.get_daily(symbol=symbol, outputsize="compact")
    sma_20_df = market_client.compute_sma(price_df, window=20)
    rsi_14_df = market_client.compute_rsi(price_df, period=14)

    return price_df, sma_20_df, rsi_14_df


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def load_treasury_snapshot():
    """
    美债收益率变化没那么快，缓存 12 小时。
    """
    settings = load_settings()
    market_client, _, _ = build_clients(settings)
    return market_client.get_treasury_10y()


@st.cache_data(ttl=7 * 24 * 60 * 60, show_spinner=False)
def load_cpi_snapshot():
    """
    CPI 是月度数据，缓存 7 天。
    """
    settings = load_settings()
    market_client, _, _ = build_clients(settings)
    return market_client.get_cpi()


def render_market_section(price_df: pd.DataFrame) -> None:
    st.subheader("价格走势")

    if price_df.empty or len(price_df) < 2:
        st.warning("价格数据不足，无法展示走势图。")
        return

    latest = price_df.iloc[-1]
    prev = price_df.iloc[-2]
    delta = latest["close"] - prev["close"]

    col1, col2, col3 = st.columns(3)
    col1.metric("最新收盘价", f"{latest['close']:.2f}", f"{delta:.2f}")
    col2.metric("最新成交量", f"{int(latest['volume']):,}")
    col3.metric("日期", latest["date"].strftime("%Y-%m-%d"))

    chart_df = price_df.tail(90).set_index("date")[["close"]]
    st.line_chart(chart_df)


def render_news_section(news_items) -> None:
    st.subheader("相关新闻")

    if not news_items:
        st.info("未抓取到新闻。")
        return

    for item in news_items:
        with st.expander(item.title):
            st.markdown(f"**来源**: {item.source or 'Unknown'}")
            st.markdown(f"**发布时间**: {item.published or 'Unknown'}")
            st.markdown(f"**链接**: {item.link}")

            if item.summary:
                st.markdown("**摘要**")
                st.write(item.summary)

            if item.content:
                st.markdown("**正文片段**")
                content_preview = item.content[:1200]
                if len(item.content) > 1200:
                    content_preview += "..."
                st.write(content_preview)


def render_analysis_section(result: dict) -> None:
    st.subheader("DeepSeek 分析结论")

    st.markdown(f"**总体判断**: `{result.get('overall_bias', 'unknown')}`")
    st.write(result.get("one_sentence_summary", ""))

    price_signal = result.get("price_signal", {})
    macro_signal = result.get("macro_signal", {})
    news_signal = result.get("news_signal", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 价格信号")
        st.json(price_signal)

    with col2:
        st.markdown("### 宏观信号")
        st.json(macro_signal)

    with col3:
        st.markdown("### 新闻信号")
        st.json(news_signal)

    st.markdown("### 风险点")
    for item in result.get("risk_points", []):
        st.write(f"- {item}")

    st.markdown("### 谨慎建议")
    st.write(result.get("action_hint", ""))


def build_market_snapshot(symbol: str) -> MarketSnapshot:
    price_df, sma_20_df, rsi_14_df = load_price_snapshot(symbol)
    treasury_10y_df = load_treasury_snapshot()
    cpi_df = load_cpi_snapshot()

    return MarketSnapshot(
        price_df=price_df,
        sma_20_df=sma_20_df,
        rsi_14_df=rsi_14_df,
        treasury_10y_df=treasury_10y_df,
        cpi_df=cpi_df,
    )


def main() -> None:
    st.sidebar.header("参数")
    symbol = st.sidebar.text_input("黄金代理标的", value="GLD")
    news_max_items = st.sidebar.slider("新闻条数", min_value=3, max_value=12, value=8)

    st.sidebar.markdown("---")
    if st.sidebar.button("清空缓存"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.sidebar.success("缓存已清空，请重新运行分析。")

    run_button = st.sidebar.button("开始分析")

    if not run_button:
        st.info("点击左侧“开始分析”以获取最新市场分析。")
        return

    try:
        settings = load_settings()
        _, news_fetcher, analyst_service = build_clients(settings)

        with st.spinner("拉取市场数据中..."):
            market = build_market_snapshot(symbol)

        with st.spinner("抓取黄金相关新闻中..."):
            news = news_fetcher.fetch_default_gold_news(max_items=news_max_items)

        with st.spinner("调用 DeepSeek 生成分析结论中..."):
            analysis_input = AnalysisInput(symbol=symbol, market=market, news=news)
            result = analyst_service.run(analysis_input)

        render_market_section(market.price_df)

        left, right = st.columns([1.1, 1.3])
        with left:
            render_news_section(news)
        with right:
            render_analysis_section(result)

    except Exception as exc:
        st.error(f"运行失败：{exc}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()