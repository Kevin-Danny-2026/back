import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# 页面基础配置
st.set_page_config(
    page_title="标普500均值回归盘后抄底决策终端",
    page_icon="📈",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #2a2e39;
    }
    .signal-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # 缓存1小时，保证获取最新数据
def fetch_spx_data():
    # 抓取标普500近5年日K线数据
    ticker = "^GSPC"
    df = yf.download(ticker, period="5y", interval="1d")
    
    # 兼容多层索引
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df[['Close', 'High', 'Low', 'Open', 'Volume']].dropna()
    
    # 计算技术指标
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # 偏离度 BIAS 200
    df['BIAS200'] = ((df['Close'] - df['SMA200']) / df['SMA200']) * 100
    
    # 偏离度 Z-Score (200日滚动标准差)
    std200 = df['Close'].rolling(window=200).std()
    df['Z_Score'] = (df['Close'] - df['SMA200']) / std200
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

# 加载数据
try:
    df = fetch_spx_data()
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    latest_close = df['Close'].iloc[-1]
    latest_sma200 = df['SMA200'].iloc[-1]
    latest_bias = df['BIAS200'].iloc[-1]
    latest_rsi = df['RSI14'].iloc[-1]
    latest_z = df['Z_Score'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    daily_change = ((latest_close - prev_close) / prev_close) * 100

    # 头部标题与更新时间
    st.title("📈 标普500 (S&P 500) 均值回归盘后抄底终端")
    st.caption(f"数据更新时间（美股盘后最新交易日）: **{latest_date}** | 数据源: Yahoo Finance")

    # --- 信号诊断逻辑 ---
    if latest_bias <= -15.0:
        signal_level = "🔴 3级抄底：极度偏离（黄金坑 / 全力部署）"
        bg_color = "#8b0000"
        action_text = "建议部署 40% 剩余抄底资金。历史统计极高胜率区间，市场处于恐慌性抛售尾声。"
    elif latest_bias <= -10.0:
        signal_level = "🟧 2级抄底：中度偏离（强力加仓）"
        bg_color = "#b8860b"
        action_text = "建议部署 40% 抄底资金。极佳盈亏比，市场出现深度洗盘。"
    elif latest_bias <= -5.0:
        signal_level = "🟡 1级抄底：轻度偏离（试探建仓）"
        bg_color = "#2e8b57"
        action_text = "建议部署 20% 试探性资金。股价回落至均线支撑区，开启第一阶梯分批介入。"
    else:
        signal_level = "🟢 0级：正常/高位区间（观望/按原计划定投）"
        bg_color = "#1e222d"
        action_text = "偏离度在安全范围内，未触发抄底阈值。维持现有仓位或常规定投即可。"

    # 顶部抄底提示 Banner
    st.markdown(
        f'<div class="signal-box" style="background-color: {bg_color}; color: white;">'
        f'{signal_level}<br><span style="font-size: 16px; font-weight: normal;">{action_text}</span>'
        f'</div>', 
        unsafe_allow_html=True
    )

    # --- 核心指标 Metrics 卡片 ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("标普500 最新收盘", f"{latest_close:,.2f}", f"{daily_change:+.2f}%")
    c2.metric("200日均线 (MA200)", f"{latest_sma200:,.2f}")
    c3.metric("MA200 偏离度 (BIAS)", f"{latest_bias:+.2f}%", help="低于 -5% 开始进入抄底观察区")
    c4.metric("RSI (14)", f"{latest_rsi:.1f}", help="<30 表示超卖")
    c5.metric("Z-Score 离差值", f"{latest_z:+.2f}", help="< -2.0 处于统计学极值下轨")

    st.markdown("---")

    # --- 交互图表绘制 (Plotly) ---
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader("📊 标普500 走势与 MA200 偏离度通道")
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08, 
            row_heights=[0.65, 0.35],
            subplot_titles=("S&P 500 价格与 200日均线", "MA200 偏离度 (%) & 抄底阈值界线")
        )

        # 1. 主图：价格与 200 日均线
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="S&P 500 收盘价", line=dict(color='#00f2fe', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA200'], name="200日均线", line=dict(color='#ff9900', width=2, dash='dot')), row=1, col=1)

        # 2. 副图：偏离度 BIAS 200
        fig.add_trace(go.Scatter(x=df.index, y=df['BIAS200'], name="偏离度 (BIAS %)", line=dict(color='#e0e0e0', width=1.2)), row=2, col=1)
        
        # 添加抄底阈值线
        fig.add_hline(y=-5, line_dash="dash", line_color="green", annotation_text="1级抄底 (-5%)", row=2, col=1)
        fig.add_hline(y=-10, line_dash="dash", line_color="orange", annotation_text="2级抄底 (-10%)", row=2, col=1)
        fig.add_hline(y=-15, line_dash="dash", line_color="red", annotation_text="3级抄底 (-15%)", row=2, col=1)
        fig.add_hline(y=0, line_color="gray", opacity=0.5, row=2, col=1)

        fig.update_layout(
            height=600,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("📜 历史触发记录 (近5年)")
        # 筛选出触发抄底信号的日期
        triggers = df[df['BIAS200'] <= -5.0][['Close', 'BIAS200', 'RSI14']].copy()
        triggers['偏离度'] = triggers['BIAS200'].apply(lambda x: f"{x:.2f}%")
        triggers['RSI'] = triggers['RSI14'].apply(lambda x: f"{x:.1f}")
        
        if not triggers.empty:
            st.dataframe(
                triggers[['Close', '偏离度', 'RSI']].sort_index(ascending=False),
                height=520,
                use_container_width=True
            )
        else:
            st.info("近5年未捕捉到偏离度 <= -5% 的记录。")

except Exception as e:
    st.error(f"数据加载失败，请检查网络或重试。错误信息: {e}")
