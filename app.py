import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from deep_translator import GoogleTranslator
from duckduckgo_search import DDGS
import streamlit.components.v1 as components
import requests # 新增：處理網路請求

# --- 頁面設定 ---
st.set_page_config(page_title="牛市股神", layout="wide")

# --- 跑馬燈邏輯 ---
def display_market_ticker():
    tickers = {
        'S&P 500': '^GSPC', '道瓊 DJI': '^DJI', '那斯達克': '^IXIC',
        '費半 SOXX': 'SOXX', '恐慌指數 VIX': '^VIX',
        'BTC': 'BTC-USD', 'ETH': 'ETH-USD', 'SOL': 'SOL-USD'
    }
    items = []
    for name, symbol in tickers.items():
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="5d")
            if len(hist) >= 2:
                p_now = hist['Close'].iloc[-1]; p_prev = hist['Close'].iloc[-2]
                change = p_now - p_prev; pct = (change/p_prev)*100
                color = "#00FF00" if change >= 0 else "#FF4B4B"
                arrow = "▲" if change >= 0 else "▼"
                items.append(f"<span style='margin-left: 30px; color: {color}; font-weight: bold; font-family: monospace; font-size: 16px;'>{name}: {p_now:,.2f} ({arrow} {pct:.2f}%)</span>")
        except: continue

    if items:
        content = "".join(items)
        ticker_html = f"""
        <style>
        .ticker-wrap {{ width: 100%; overflow: hidden; background-color: #0E1117; border-bottom: 1px solid #303030; white-space: nowrap; padding: 8px 0; }}
        .ticker {{ display: inline-block; animation: marquee 60s linear infinite; }}
        .ticker-wrap:hover .ticker {{ animation-play-state: paused; }}
        @keyframes marquee {{ 0% {{ transform: translate(100%, 0); }} 100% {{ transform: translate(-100%, 0); }} }}
        </style>
        <div class="ticker-wrap"><div class="ticker">{content} {content} {content}</div></div>
        """
        st.markdown(ticker_html, unsafe_allow_html=True)
    else: st.warning("正在連線市場數據...")

display_market_ticker()

# --- 主標題 ---
st.title("🏹 美股健康檢查室")

# --- 初始化 Session State ---
if 'analyzed' not in st.session_state: st.session_state.analyzed = False
if 'ticker' not in st.session_state: st.session_state.ticker = "TSM"

# --- 側邊欄 ---
with st.sidebar:
    st.header("鎖定目標")
    with st.form(key='sniper_form'):
        ticker_input = st.text_input("輸入美股代號", value=st.session_state.ticker)
        run_btn = st.form_submit_button("開始分析")

    if run_btn:
        st.session_state.analyzed = True
        st.session_state.ticker = ticker_input.upper() if ticker_input else None

    st.markdown("### 🔥 熱門市場標的")
    hot_tickers = ['NVDA', 'TSM', 'AAPL', 'TSLA', 'GOOGL', 'AMZN', 'MSFT', 'META', 'SNDK']
    cols = st.columns(3)
    for i, hot_t in enumerate(hot_tickers):
        if cols[i % 3].button(hot_t, use_container_width=True):
            st.session_state.ticker = hot_t
            st.session_state.analyzed = True
            st.rerun() # 點擊後立即重新載入分析

    st.markdown("---")
    st.info("""
    💡 **評分標準 (總分 10 分)**
    **🚀 成長動能 (4分)**: 收益修正, 獲利驚喜, 營收成長, 獲利成長
    **🏰 獲利分析 (4分)**: 毛利率, 淨利率, ROE, 利潤趨勢
    **🛡️ 財務健康 (2分)**: 現金流量, 負債比
    """)
    if st.session_state.analyzed and st.session_state.ticker:
        nasdaq_url = f"https://www.nasdaq.com/market-activity/stocks/{st.session_state.ticker.lower()}/financials"
        st.link_button(f"前往 Nasdaq 驗證 {st.session_state.ticker}", nasdaq_url)

# --- 數據抓取 ---
@st.cache_data(ttl=3600)
def get_company_profile(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        inst_pct = info.get('heldPercentInstitutions', 0)
        insider_pct = info.get('heldPercentInsiders', 0)
        targets = {
            'current': info.get('currentPrice'), 'low': info.get('targetLowPrice'),
            'high': info.get('targetHighPrice'), 'mean': info.get('targetMeanPrice'),
            'count': info.get('numberOfAnalystOpinions')
        }
        return info, inst_pct, insider_pct, targets
    except: return None, 0, 0, {}

@st.cache_data(ttl=3600)
def get_market_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        return stock.history(period="1y", interval="1d"), stock.history(period="2y", interval="1wk")
    except: return None, None

@st.cache_data(ttl=3600)
def get_financial_data(symbol):
    stock = yf.Ticker(symbol)
    try:
        # 取最後 5 季以便計算 YoY (本期 vs 4季前)
        return stock.quarterly_financials.T.sort_index().tail(5), \
               stock.quarterly_balance_sheet.T.sort_index().tail(5), \
               stock.quarterly_cashflow.T.sort_index().tail(5)
    except: return None

@st.cache_data(ttl=3600)
def get_estimates_data(symbol):
    stock = yf.Ticker(symbol)
    rev_score = 0; sur_score = 0; sur_text = "N/A"
    try:
        upgrades = stock.upgrades_downgrades
        if upgrades is not None and not upgrades.empty:
            recent = upgrades[upgrades.index > (pd.Timestamp.now() - pd.DateOffset(months=3))]
            bullish = recent[(recent['Action'] == 'Up') | (recent['ToGrade'].str.contains('Buy|Outperform', case=False, regex=True))]
            if len(bullish) > 0: rev_score = 1
    except: pass
    try:
        earn = stock.earnings_dates
        if earn is not None and not earn.empty:
            valid = earn[earn['Reported EPS'].notna()].iloc[0]
            if valid['Reported EPS'] > valid['EPS Estimate']: sur_score = 1; sur_text = "Beat"
            else: sur_text = "Miss"
    except: pass
    return rev_score, sur_score, sur_text

# --- 新聞抓取 ---
@st.cache_data(ttl=3600)
def get_news_data(symbol):
    results = []
    try:
        with DDGS() as ddgs:
            keywords = f"{symbol} stock news"
            ddg_news = list(ddgs.news(keywords=keywords, max_results=15))
            if ddg_news:
                results = sorted(
                    ddg_news,
                    key=lambda x: pd.to_datetime(x.get('date'), errors='coerce') or pd.Timestamp.min,
                    reverse=True
                )
    except Exception as e:
        print(f"News Error: {e}")
    return results

def translate_text(text):
    try: return GoogleTranslator(source='auto', target='zh-TW').translate(text) if text else ""
    except: return text

@st.cache_data(ttl=3600)
def get_benchmark_data(benchmark_symbol, period="1y", interval="1d"):
    try: return yf.Ticker(benchmark_symbol).history(period=period, interval=interval)['Close']
    except: return None

# --- 定義數據安全工具箱 (解決 NameError 的關鍵) ---

def safe_get(df, col):
    """安全獲取最新的財務數值"""
    if df is not None and col in df.columns and not df[col].empty:
        return df[col].iloc[-1]
    return 0

def safe_yoy_growth(df, col):
    """計算年度增長率 (YoY)，解決數據不足 5 季的問題"""
    try:
        if df is not None and col in df.columns and len(df) >= 5:
            now = df[col].iloc[-1]
            last_year = df[col].iloc[-5]
            if last_year != 0:
                return (now - last_year) / abs(last_year)
        return 0
    except:
        return 0

def safe_growth(df, col):
    """計算季度增長率 (QoQ)，解決 image_6d4a6e 報錯問題"""
    try:
        if df is not None and col in df.columns and len(df) >= 2:
            now = df[col].iloc[-1]
            prev = df[col].iloc[-2]
            if prev != 0:
                return (now - prev) / abs(prev)
        return 0
    except:
        return 0

def calculate_technical_indicators(df, is_weekly=False):
    if df.empty: return df
    df = df.copy()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    if not is_weekly:
        v = df['Volume'].values; tp = (df['High'] + df['Low'] + df['Close']) / 3
        df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    return df

# --- 繪圖 ---
def plot_holdings_pie(inst_pct, insider_pct):
    if inst_pct < 1: inst_pct *= 100
    if insider_pct < 1: insider_pct *= 100
    public_pct = max(0, 100 - inst_pct - insider_pct)
    labels = ['機構', '內部人/股東', '大眾/其他']
    values = [inst_pct, insider_pct, public_pct]
    colors = ['#FF4B4B', '#FFA15A', '#606060']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker=dict(colors=colors), textinfo='percent+label')])
    fig.update_layout(title="持股結構", template="plotly_dark", height=300, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- 3. 分析師預測繪圖函數 (根據 image_1220bd 修改並加入百分比) ---
def plot_analyst_forecast(hist_df, targets):
    if hist_df is None or hist_df.empty or not targets.get('mean'):
        return go.Figure()
    
    # 取得現價
    curr = targets.get('current', hist_df['Close'].iloc[-1])
    mean, high, low = targets.get('mean'), targets.get('high'), targets.get('low')
    last_date = hist_df.index[-1]
    future_date = last_date + timedelta(days=365)
    
    # 計算漲跌幅百分比 (新增功能)
    def get_pct(target_price):
        return ((target_price - curr) / curr) * 100

    fig = go.Figure()
    
    # 歷史走勢線
    fig.add_trace(go.Scatter(x=hist_df.index, y=hist_df['Close'], mode='lines', name='歷史', line=dict(color='#1E90FF', width=2)))
    
    # 最高目標 (含百分比)
    if high:
        pct = get_pct(high)
        fig.add_trace(go.Scatter(x=[last_date, future_date], y=[curr, high], mode='lines+markers+text', 
                                 name='最高', line=dict(color='#00CC96', width=2, dash='dot'),
                                 text=[None, f"${high} ({pct:+.1f}%)"], textposition="top right"))
    
    # 平均目標 (含百分比)
    if mean:
        pct = get_pct(mean)
        fig.add_trace(go.Scatter(x=[last_date, future_date], y=[curr, mean], mode='lines+markers+text', 
                                 name='平均', line=dict(color='white', width=2, dash='dash'),
                                 text=[None, f"${mean} ({pct:+.1f}%)"], textposition="middle right"))
        
    # 最低目標 (含百分比)
    if low:
        pct = get_pct(low)
        fig.add_trace(go.Scatter(x=[last_date, future_date], y=[curr, low], mode='lines+markers+text', 
                                 name='最低', line=dict(color='#EF553B', width=2, dash='dot'),
                                 text=[None, f"${low} ({pct:+.1f}%)"], textposition="bottom right"))

    fig.add_trace(go.Scatter(x=[last_date], y=[curr], mode='markers', marker=dict(color='white', size=8), showlegend=False))
    
    fig.update_layout(title=f"分析師目標價 ({targets.get('count', 'N/A')}位)", template="plotly_dark", height=400, margin=dict(l=20, r=50, t=50, b=20))
    return fig

def plot_financial_charts(q_inc):
    dates = q_inc.index.strftime('%Y-%m')
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    if 'Total Revenue' in q_inc.columns: fig1.add_trace(go.Bar(x=dates, y=q_inc['Total Revenue'], name="營收", marker_color='#1f77b4', opacity=0.7), secondary_y=False)
    if 'Net Income' in q_inc.columns: fig1.add_trace(go.Scatter(x=dates, y=q_inc['Net Income'], name="淨利", line=dict(color='#ff7f0e', width=3)), secondary_y=True)
    fig1.update_layout(title="營收與淨利", template="plotly_dark", height=350, margin=dict(l=20, r=20, t=40, b=20))

    fig2 = go.Figure()
    if 'Basic EPS' in q_inc.columns: fig2.add_trace(go.Bar(x=dates, y=q_inc['Basic EPS'], name="EPS", marker_color=['#00CC96' if v>=0 else '#EF5350' for v in q_inc['Basic EPS']]))
    fig2.update_layout(title="EPS 趨勢", template="plotly_dark", height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig1, fig2

def plot_margin_trends(q_inc):
    dates = q_inc.index.strftime('%Y-%m')
    fig = go.Figure()
    if 'Total Revenue' in q_inc.columns:
        rev = q_inc['Total Revenue']
        if 'Gross Profit' in q_inc.columns: fig.add_trace(go.Scatter(x=dates, y=q_inc['Gross Profit']/rev*100, name="毛利率", line=dict(color='#00CC96')))
        if 'Operating Income' in q_inc.columns: fig.add_trace(go.Scatter(x=dates, y=q_inc['Operating Income']/rev*100, name="營益率", line=dict(color='#FFA15A')))
        if 'Net Income' in q_inc.columns: fig.add_trace(go.Scatter(x=dates, y=q_inc['Net Income']/rev*100, name="淨利率", line=dict(color='#EF553B')))
    fig.update_layout(title="三率走勢", template="plotly_dark", height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def plot_extra_financials(q_bal, q_cash):
    dates = q_bal.index.strftime('%Y-%m')
    fig_bs = go.Figure()
    if 'Total Assets' in q_bal.columns: fig_bs.add_trace(go.Bar(x=dates, y=q_bal['Total Assets'], name='總資產', marker_color='#1f77b4'))
    liab = 'Total Liabilities Net Minority Interest' if 'Total Liabilities Net Minority Interest' in q_bal.columns else 'Total Liabilities'
    if liab in q_bal.columns: fig_bs.add_trace(go.Bar(x=dates, y=q_bal[liab], name='總債務', marker_color='#EF553B'))
    fig_bs.update_layout(title="資產負債結構", template="plotly_dark", height=350, barmode='group', margin=dict(l=20, r=20, t=40, b=20))

    fig_cf = go.Figure()
    if 'Operating Cash Flow' in q_cash.columns: fig_cf.add_trace(go.Scatter(x=dates, y=q_cash['Operating Cash Flow'], name='營運現金流', fill='tozeroy', line=dict(color='#00CC96')))
    if 'Capital Expenditure' in q_cash.columns: fig_cf.add_trace(go.Bar(x=dates, y=q_cash['Capital Expenditure'], name='資本支出', marker_color='#EF553B'))
    fig_cf.update_layout(title="現金流向", template="plotly_dark", height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig_bs, fig_cf

def plot_technical_chart(df, ticker, period_name="日線", benchmarks=None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2], subplot_titles=(f'{ticker} {period_name}', 'Volume', 'MACD'))
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], mode='lines', name='MA200', line=dict(color='white', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='#FF69B4', width=1.5)), row=1, col=1)
    if 'VWAP' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='#90EE90', width=1.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', line=dict(color='#1E90FF', width=1), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', line=dict(color='#1E90FF', width=1), fill='tonexty', fillcolor='rgba(30,144,255,0.1)', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], mode='lines', name='BB中線', line=dict(color='orange', width=1)), row=1, col=1)

    if benchmarks:
        start = df['Close'].iloc[0]; colors = {'SPY':'#FFFF00', 'SOXX':'#00FFFF', '^DJI':'#FF00FF', '^IXIC':'#ADFF2F'}
        for n, d in benchmarks.items():
            if d is not None:
                aligned = d[df.index[0]:]
                if not aligned.empty: fig.add_trace(go.Scatter(x=aligned.index, y=aligned*(start/aligned.iloc[0]), mode='lines', name=f'vs {n}', line=dict(color=colors.get(n,'gray'), width=2), opacity=0.8), row=1, col=1)

    colors = ['#00CC96' if r['Close']>=r['Open'] else '#EF553B' for i,r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#2962FF', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='#FF6D00', width=1.5)), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Hist', marker_color=['#26A69A' if v>=0 else '#EF5350' for v in df['MACD_Hist']]), row=3, col=1)
    fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.01, x=0))
    return fig

def generate_strategy(score, current_price, ma50):
    holder_advice = ""
    if score >= 7:
        if current_price > ma50: holder_advice = "🚀 **續抱 (Hold)**：基本面強勁且趨勢向上，為核心持股。"
        else: holder_advice = "🛡️ **觀察 (Watch)**：體質優良但股價回檔，未破長期支撐前不輕易賣出。"
    elif score >= 5:
        if current_price > ma50: holder_advice = "⚠️ **續抱但謹慎**：留意技術面變化，嚴設停損。"
        else: holder_advice = "✂️ **減碼/出場**：優勢不再，換股操作。"
    else: holder_advice = "🏃 **趁反彈離場**：基本面與技術面雙弱。"

    buyer_advice = ""
    if score >= 7:
        if current_price > ma50: buyer_advice = "💰 **買進 (Buy)**：等待回測 MA50 或布林中線進場。"
        else: buyer_advice = "👀 **等待 (Wait)**：等待股價止穩並站回生命線，優質股的黃金買點。"
    elif score >= 5:
        if current_price > ma50: buyer_advice = "🤔 **短線操作**：僅適合技術面操作。"
        else: buyer_advice = "⛔ **觀望**：目前缺乏催化劑。"
    else: buyer_advice = "⛔ **遠離 (Avoid)**。"

    return holder_advice, buyer_advice

# --- 主程式 ---
if st.session_state.analyzed and st.session_state.ticker:
    ticker = st.session_state.ticker
    tab1, tab2, tab3, tab4 = st.tabs(["🏢 公司簡介", "📰 市場輿情", "📊 財報 & 評分", "📈 雙週期走勢 & 戰術"])

    with st.spinner(f"正在全速運算 {ticker} ..."):
        profile, inst_pct, insider_pct, targets = get_company_profile(ticker)
        hist_d, hist_w = get_market_data(ticker)
        fin_data = get_financial_data(ticker)
        news_data = get_news_data(ticker)

        if profile is None or fin_data is None:
            st.error("查無數據，請確認代號。"); st.session_state.analyzed = False; st.stop()

        q_inc, q_bal, q_cash = fin_data
        rev_score, sur_score, sur_text = get_estimates_data(ticker)

        with tab1:
            st.header(f"{ticker} - {profile.get('longName','')}")
            c1, c2, c3 = st.columns(3)
            c1.info(f"板塊: {profile.get('sector','')}"); c2.info(f"產業: {profile.get('industry','')}"); c3.info(f"員工: {profile.get('fullTimeEmployees','')}")
            c_t, c_p = st.columns([2,1])
            with c_t:
                with st.expander("📝 業務概覽", True): st.write(translate_text(profile.get('longBusinessSummary','')))
            with c_p: st.plotly_chart(plot_holdings_pie(inst_pct, insider_pct), use_container_width=True)
            st.markdown("---"); st.subheader("🎯 分析師目標價"); st.plotly_chart(plot_analyst_forecast(hist_d, targets), use_container_width=True)

        with tab2: # 輿情
            st.header(f"📰 {ticker} 近期市場輿情")
            if news_data:
                for item in news_data:
                    with st.container():
                        st.subheader(translate_text(item.get('title','')))
                        st.caption(f"來源: {item.get('source','')} | 時間: {item.get('date','')}")
                        st.markdown(f"**摘要**: {translate_text(item.get('body',''))}")
                        if item.get('url') or item.get('href'): st.markdown(f"[閱讀全文]({item.get('url') or item.get('href')})")
                        st.divider()
            else: st.info("暫無新聞數據")

        with tab3:
            st.subheader("📊 財務報表視覺化")
            f_inc, f_eps = plot_financial_charts(q_inc)
            f_mar = plot_margin_trends(q_inc)
            c_g1, c_g2 = st.columns(2)
            c_g1.plotly_chart(f_inc, use_container_width=True)
            c_g2.plotly_chart(f_mar, use_container_width=True)
            st.plotly_chart(f_eps, use_container_width=True)
            f_bs, f_cf = plot_extra_financials(q_bal, q_cash)
            c_g3, c_g4 = st.columns(2)
            c_g3.plotly_chart(f_bs, use_container_width=True)
            c_g4.plotly_chart(f_cf, use_container_width=True)

            st.markdown("---")
            st.subheader("🏆 加權評分 (滿分10)")

         # --- 評分計算 (不簡化版：修復變數名稱、函數並適應金融業) ---

# 1. 自動偵測可用的利潤指標 (解決金融業 Operating Income 缺失問題)
available_cols = q_inc.columns.tolist()
if 'Operating Income' in available_cols and q_inc['Operating Income'].iloc[-1] != 0:
    profit_col = 'Operating Income'
    profit_label = "營益率"
else:
    # 金融業自動改採「淨利」計算利潤趨勢
    profit_col = 'Net Income'
    profit_label = "淨利率"

# 2. 核心數值獲取
rev_now = safe_get(q_inc, 'Total Revenue')
rev_g_yoy = safe_yoy_growth(q_inc, 'Total Revenue')

# 動態獲取利潤值
op_inc_now = safe_get(q_inc, profit_col)
op_margin_now = op_inc_now / rev_now if rev_now else 0

# 計算前一期利潤率用於 QoQ 對比
if len(q_inc) >= 2:
    prev_rev = q_inc.iloc[-2]['Total Revenue'] if 'Total Revenue' in q_inc.columns else 0
    prev_profit = q_inc.iloc[-2][profit_col] if profit_col in q_inc.columns else 0
    op_margin_prev = prev_profit / prev_rev if prev_rev else 0
else:
    op_margin_prev = 0

# 獲取其他財務指標
gross_margin = safe_get(q_inc, 'Gross Profit') / rev_now if rev_now else 0
net_income = safe_get(q_inc, 'Net Income')
net_margin = net_income / rev_now if rev_now else 0
total_equity = safe_get(q_bal, 'Stockholders Equity')
total_debt = safe_get(q_bal, 'Total Debt')
debt_to_equity = total_debt / total_equity if total_equity else 999
fcf = safe_get(q_cash, 'Operating Cash Flow') + safe_get(q_cash, 'Capital Expenditure')
eps_g_qoq = safe_growth(q_inc, 'Basic EPS')  #
roe = (net_income / total_equity) * 100 if total_equity else 0

# 3. 執行加權評分 (總分 10 分)
score = 0
res = []

# [成長動能]
p = 1.0 if rev_score else 0; score += p
res.append(["收益修正", p, "1.0", "有" if p else "無", "分析師看多"])

p = 1.0 if sur_score >= 1 else 0; score += p
res.append(["獲利驚喜", p, "1.0", sur_text, "Beat預期"])

p = 1.0 if rev_g_yoy > 0.20 else (0.5 if rev_g_yoy > 0.10 else 0); score += p
res.append(["營收成長", p, "1.0", f"{rev_g_yoy:.1%}", "YoY成長"])

p = 1.0 if eps_g_qoq > 0.15 else (0.5 if eps_g_qoq > 0.05 else 0); score += p
res.append(["獲利成長", p, "1.0", f"{eps_g_qoq:+.1%}", "QoQ成長"])

# [獲利分析]
p = 1.0 if gross_margin > 0.50 else (0.5 if gross_margin > 0.30 else 0); score += p
res.append(["毛利率", p, "1.0", f"{gross_margin:.1%}", "定價能力"])

p = 1.0 if net_margin > 0.20 else (0.5 if net_margin > 0.10 else 0); score += p
res.append(["淨利率", p, "1.0", f"{net_margin:.1%}", "獲利體質"])

p = 1.0 if roe > 20 else (0.5 if roe > 15 else 0); score += p
res.append(["ROE", p, "1.0", f"{roe:.1f}%", "股東權益"])

# 動態判斷利潤趨勢 (金融業會自動顯示淨利率)
p = 1.0 if op_margin_now > op_margin_prev else 0; score += p
res.append([f"利潤趨勢({profit_label})", p, "1.0", f"{op_margin_now:.1%}", "QoQ擴大" if p else "QoQ縮減"])

# [財務健康]
p = 1.0 if fcf > 0 else 0; score += p
res.append(["現金流量", p, "1.0", f"${fcf/1e6:,.0f}M", "自由現金流"])

p = 1.0 if debt_to_equity < 0.8 else (0.5 if debt_to_equity < 2.0 else 0); score += p
res.append(["負債比", p, "1.0", f"{debt_to_equity:.2f}", "財務槓桿"])

# 4. 輸出評分表格
c_sc, c_dt = st.columns([1, 2])
            with c_sc:

                st.metric("總分", f"{score:.1f} / 10")

                if score>=7: st.success("🟢 強烈推薦")

                elif score>=4: st.warning("🟡 持有")

                else: st.error("🔴 賣出")

            with c_dt:

                st.dataframe(pd.DataFrame(res, columns=["指標","得分","權重","數據","評註"]), use_container_width=True, hide_index=True)

        with tab4: # 走勢
            df_daily = calculate_technical_indicators(hist_d, False)
            hold, buy = generate_strategy(score, df_daily['Close'].iloc[-1], df_daily['MA50'].iloc[-1])
            st.markdown("### 🧠 操作建議"); c_h, c_b = st.columns(2); c_h.info(f"持有者: {hold}"); c_b.success(f"空手者: {buy}")

            with st.expander("⚙️ 疊加大盤"):
                c1, c2, c3, c4 = st.columns(4)
                s_spy = c1.checkbox("疊加標普500 (SPY)")
                show_soxx = c2.checkbox("疊加費半 (SOXX)")
                show_dji = c3.checkbox("疊加道瓊 (DJI)")
                show_ixic = c4.checkbox("疊加納指 (IXIC)")

            benchs_d = {}
            def fetch(s): return get_benchmark_data(s, "1y", "1d")

            if s_spy: benchs_d['SPY'] = fetch('SPY')
            if show_soxx: benchs_d['SOXX'] = fetch('SOXX')
            if show_dji: benchs_d['^DJI'] = fetch('^DJI')
            if show_ixic: benchs_d['^IXIC'] = fetch('^IXIC')

            t1, t2 = st.tabs(["日線圖", "週線圖"])
            with t1:
                st.plotly_chart(plot_technical_chart(df_daily, ticker, "日線", benchs_d), use_container_width=True)
            with t2:
                st.plotly_chart(plot_technical_chart(calculate_technical_indicators(hist_w, True), ticker, "週線"), use_container_width=True)
else:
    st.info("👈 請輸入代碼")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📖 使用說明")
        st.markdown("""
        1. **快速搜尋**：在左側搜尋框輸入美股代號（例如：AAPL, TSLA）。
        2. **熱門標的**：直接點擊左側「🔥 熱門市場標的」按鈕快速分析。
        3. **四大分頁**：
            - **🏢 公司簡介**：了解企業業務範圍。
            - **📰 市場輿情**：查看最新的相關新聞與趨勢。
            - **📊 財報 & 評分**：檢查公司的獲利能力與財務健康度。
            - **📈 雙週期走勢 & 戰術**：結合技術指標給予操作建議。
        4. **如何在手機端使用**：
            -  iOS (Safari 瀏覽器):
            1. 進入 https://5f4cx8cawucvqrc42s6o6q.streamlit.app/
            2. 點擊瀏覽器底部的 **「分享」** 圖示 (方框箭頭朝上)。
            3. 往下滑動找到並點擊 **「加入主畫面」**。
            4. 點擊右上角的 **「新增」**，桌面就會出現專屬圖示！
            -  Android (Chrome 瀏覽器):
            1. 進入 https://5f4cx8cawucvqrc42s6o6q.streamlit.app/
            2. 點擊瀏覽器右上角的 **「三個點」** 選單。
            3. 選擇 **「安裝應用程式」** 或 **「將網頁加入主畫面」**。
            4. 點擊 **「新增」** 後，即可在手機桌面一鍵啟動！
            - **💡 小撇步**: 加入主畫面後，操作起來會像真正的 App 一樣全螢幕運行，體驗更順暢喔！
        5. 如何在電腦端使用：
            - ** 永久保存 https://5f4cx8cawucvqrc42s6o6q.streamlit.app/**:
            """)
    with col2:
        st.subheader("📜 更新日誌")
        st.markdown("""
        - **v14.0(更新進行中)**：新增Gemini作為投資助理。
        - **v13.10**：於分析師預測價旁標註出潛在漲跌幅空間百分比。(2026/01/08)
        - **v13.9**：新增首頁使用說明與更新日誌。
        - **v13.8**：側邊欄新增「熱門市場標的」快速點擊按鈕。
        """)
