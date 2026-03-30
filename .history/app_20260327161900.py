"""
参考仓库 / 文档：
- GitHub: KVignesh122/AssetNewsSentimentAnalyzer
  用途：借鉴“新闻驱动的资产分析工具”定位
- GitHub: alphavantage/alpha_vantage_mcp
  用途：借鉴黄金/宏观数据范围
- DeepSeek API Docs
  用途：DeepSeek OpenAI-compatible 接法
"""

from __future__ import annotations

import traceback

import pandas as pd
import streamlit as st

from config import Settings
from data.market_data import AlphaVantageClient
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
    market_client = AlphaVantageClient(settings.alpha_vantage_api_key, timeout=settings.request_timeout_sec)
    news_fetcher = GoldNewsFetcher(timeout=settings.article_timeout_sec)
    llm_client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
    )
    analyst_service = GoldAnalystService(llm_client)
    return market_client, news_fetcher, analyst_service


def render_market_section(price_df: pd.DataFrame) -> None:
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
                st.write(item.content[:1200] + ("..." if len(item.content) > 1200 else ""))


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
    
@st.cache_data(ttl= 60 * 60)
def load_market_snapshot(symbol: str):
    settings = load_settings()
    market_client, _ , _ = build_clients(settings)
    return market_client.get_market_snapshot(symbol=symbol)
    


def main() -> None:
    st.sidebar.header("参数")
    symbol = st.sidebar.text_input("黄金代理标的", value="GLD")
    news_max_items = st.sidebar.slider("新闻条数", min_value=3, max_value=12, value=8)

    run_button = st.sidebar.button("开始分析")

    if not run_button:
        st.info("点击左侧“开始分析”以获取最新市场分析。")
        return

    try:
        settings = load_settings()
        market_client, news_fetcher, analyst_service = build_clients(settings)

        with st.spinner("拉取市场数据中..."):
            # market = market_client.get_market_snapshot(symbol=symbol)
            market = load_market_snapshot(symbol)

        with st.spinner("抓取黄金相关新闻中..."):
            news = news_fetcher.fetch_default_gold_news(max_items=news_max_items)

        with st.spinner("调用 DeepSeek 生成分析结论中..."):
            analysis_input = AnalysisInput(symbol=symbol, market=market, news=news)
            result = analyst_service.run(analysis_input)

        st.subheader("价格走势")
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