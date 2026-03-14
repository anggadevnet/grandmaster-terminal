import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import numpy as np
import ccxt
import time

# --- 1. PAGE CONFIG (White Clean Theme) ---
st.set_page_config(layout="wide", page_title="SniperPalmerah - Strategist", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; color: #333; }
    div[data-testid="stMetric"] {
        background-color: #F8F9FA; border: 1px solid #E9ECEF; border-radius: 12px; padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stMetric label { color: #495057; font-size: 13px; font-weight: 600; }
    .stMetric value { color: #212529; font-size: 28px; font-weight: 800; }
    .stSidebar { background-color: #F8F9FA; border-right: 1px solid #E9ECEF; }
    
    /* Custom Alert Boxes */
    .alert-buy { background-color: #e6fffa; border-left: 5px solid #00C853; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .alert-sell { background-color: #ffebee; border-left: 5px solid #FF1744; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
    .alert-wait { background-color: #f5f5f5; border-left: 5px solid #9E9E9E; padding: 15px; border-radius: 5px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 2. HELPER FUNCTIONS ---

def get_best_sensitivity(tf):
    mapping = {'1m': 3, '5m': 4, '15m': 5, '1h': 6, '4h': 8, '1d': 10}
    return mapping.get(tf, 8)

@st.cache_data(ttl=1800)
def get_symbols(ex_id):
    try:
        ex = getattr(ccxt, ex_id)({'enableRateLimit': True, 'timeout': 15000})
        ex.load_markets()
        return sorted([s for s, m in ex.markets.items() if m.get('type') == 'spot' and m.get('quote') == 'USDT' and m.get('active')])
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return ["BTC/USDT"]

def fetch_data(symbol, tf, limit=500, ex_id='binance'):
    try:
        ex = getattr(ccxt, ex_id)({'enableRateLimit': True, 'timeout': 15000})
        if tf not in ex.timeframes:
            if tf == '6h': tf = '4h'
        data = ex.fetch_ohlcv(symbol, tf, limit=limit)
        df = pd.DataFrame(data, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        return df
    except: return pd.DataFrame()

# --- 3. TECHNICAL & PATTERN CALCULATOR ---

def calculate_all(df):
    if df.empty: return df
    df = df.copy()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200) if len(df) > 200 else np.nan
    df['Vol_MA'] = df['Volume'].rolling(20).mean()
    df['RVOL'] = df['Volume'] / df['Vol_MA']
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Candlestick Patterns (using pandas_ta internal logic manually simplified for streamlit)
    # Hammer: Open == High, Close == High, Long Lower Wick
    df['Hammer'] = (((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close'])) & 
                   ((df['Close'] - df['Low']) / (.001 + df['High'] - df['Low']) > 0.6) & 
                   ((df['Open'] - df['Low']) / (.001 + df['High'] - df['Low']) > 0.6)).astype(int)
                   
    # Bullish Engulfing
    df['Bull_Engulf'] = ((df['Close'] > df['Open'].shift(1)) & 
                         (df['Open'] < df['Close'].shift(1)) & 
                         (df['Close'] > df['Open'])).astype(int)
    return df

def get_smc_structure(df, order=10):
    signals = []
    struct = {'trend': 'neutral', 'last_high': 0, 'last_low': 0}
    
    highs = argrelextrema(df['High'].values, np.greater, order=order)[0]
    lows = argrelextrema(df['Low'].values, np.less, order=order)[0]
    
    if len(highs) > 0: struct['last_high'] = df['High'].iloc[highs[-1]]
    if len(lows) > 0: struct['last_low'] = df['Low'].iloc[lows[-1]]
    
    last_close = df['Close'].iloc[-1]
    if len(highs) >= 2:
        prev_high = df['High'].iloc[highs[-2]]
        if last_close > prev_high and df['Close'].iloc[-2] <= prev_high:
            signals.append({'time': df.index[-1], 'price': prev_high, 'type': 'BOS Bull', 'text': 'BOS ↑'})

    if len(lows) >= 2:
        prev_low = df['Low'].iloc[lows[-2]]
        if last_close < prev_low and df['Close'].iloc[-2] >= prev_low:
            signals.append({'time': df.index[-1], 'price': prev_low, 'type': 'CHoCH Bear', 'text': 'CHoCH ↓'})
            
    return signals, struct

def get_order_blocks(df, lookback=20):
    obs = []
    try:
        lows = argrelextrema(df['Low'].values, np.less, order=5)[0]
        if len(lows) > 0:
            idx = lows[-1]
            search = df.iloc[max(0, idx-lookback):idx]
            bear = search[search['Close'] < search['Open']]
            if not bear.empty:
                ob_candle = bear.iloc[-1]
                obs.append({'type': 'Bull', 'time': bear.index[-1], 'h': ob_candle['High'], 'l': ob_candle['Low']})

        highs = argrelextrema(df['High'].values, np.greater, order=5)[0]
        if len(highs) > 0:
            idx = highs[-1]
            search = df.iloc[max(0, idx-lookback):idx]
            bull = search[search['Close'] > search['Open']]
            if not bull.empty:
                ob_candle = bull.iloc[-1]
                obs.append({'type': 'Bear', 'time': bull.index[-1], 'h': ob_candle['High'], 'l': ob_candle['Low']})
    except: pass
    return obs

# --- NEW ANALYSIS FUNCTIONS ---

def analyze_fib(df, lookback=100):
    """Menghitung level Fibonacci Retracement"""
    recent = df.iloc[-lookback:]
    high = recent['High'].max()
    low = recent['Low'].min()
    diff = high - low
    
    # Asumsi trend naik jika close terakhir > low + (range/2)
    trend = "Up" if df['Close'].iloc[-1] > (low + diff/2) else "Down"
    
    levels = {}
    if trend == "Up": # Retracement dari bawah ke atas
        levels['0.0 (Low)'] = low
        levels['0.236'] = low + (diff * 0.236)
        levels['0.382'] = low + (diff * 0.382)
        levels['0.5'] = low + (diff * 0.5)
        levels['0.618 (Golden)'] = low + (diff * 0.618)
        levels['0.786'] = low + (diff * 0.786)
        levels['1.0 (High)'] = high
    else: # Retracement dari atas ke bawah
        levels['1.0 (High)'] = high
        levels['0.786'] = high - (diff * 0.236) # Kebalik untuk visual
        levels['0.618 (Golden)'] = high - (diff * 0.382)
        levels['0.5'] = high - (diff * 0.5)
        levels['0.382'] = high - (diff * 0.618)
        levels['0.236'] = high - (diff * 0.786)
        levels['0.0 (Low)'] = low
        
    return levels, high, low

def analyze_double_bottom(df, lookback=50):
    """Deteksi Double Bottom dalam 50 candle terakhir"""
    recent = df.iloc[-lookback:]
    # Cari local lows
    lows_idx = argrelextrema(recent['Low'].values, np.less, order=3)[0]
    
    if len(lows_idx) >= 2:
        # Ambil 2 low terakhir
        l1_idx = lows_idx[-1]
        l2_idx = lows_idx[-2]
        
        p1 = recent['Low'].iloc[l1_idx]
        p2 = recent['Low'].iloc[l2_idx]
        
        # Syarat Double Bottom: Harga mirip (diff < 1%), dan ada gap waktu
        if abs(p1 - p2) / p1 < 0.01 and l1_idx != l2_idx:
            # Syarat Konfirmasi: Harga close sekarang sudah di atas leher (neckline)
            # Neckline kira-kira adalah high di antara dua low tersebut
            between_highs = recent['High'].iloc[l2_idx:l1_idx]
            if not between_highs.empty:
                neckline = between_highs.max()
                if recent['Close'].iloc[-1] > neckline:
                    return True, neckline
    
    return False, 0

def analyze_hidden_divergence(df):
    """Deteksi Hidden Bullish Divergence: Price Higher Low, RSI Lower Low"""
    if len(df) < 20: return False
    
    # Ambil 2 low terakhir
    lows_idx = argrelextrema(df['Low'].values, np.less, order=3)[0]
    if len(lows_idx) >= 2:
        curr_idx = lows_idx[-1]
        prev_idx = lows_idx[-2]
        
        # Price Higher Low
        p_curr = df['Low'].iloc[curr_idx]
        p_prev = df['Low'].iloc[prev_idx]
        
        # RSI Lower Low
        r_curr = df['RSI'].iloc[curr_idx]
        r_prev = df['RSI'].iloc[prev_idx]
        
        # Hidden Bullish: Price naik (Higher Low), Tapi RSI turun (Lower Low) -> Ini tanda kuat trend continuation
        if p_curr > p_prev and r_curr < r_prev:
            return True
            
    return False

def analyze_reversal_candles(df):
    """Deteksi candle pembalikan di candle terakhir"""
    last = df.iloc[-1]
    patterns = []
    
    if last['Hammer'] == 1: patterns.append("Hammer 🔨")
    if last['Bull_Engulf'] == 1: patterns.append("Bull Engulf 🐂")
    
    # Doji Detector (Open hampir sama dengan Close)
    body = abs(last['Close'] - last['Open'])
    range_candle = last['High'] - last['Low']
    if range_candle > 0 and body/range_candle < 0.1: patterns.append("Doji ⚖️")
    
    return patterns

# --- 4. CONFLUENCE ENGINE (UPDATED) ---

def detailed_analysis(df, struct, obs, fib_levels):
    details = []
    score = 0
    last = df.iloc[-1]
    
    # 1. Trend
    if last['EMA_20'] > last['EMA_50']:
        score += 1; details.append({"Factor": "Trend", "Status": "Up ↑", "Impact": "+1", "Color": "green"})
    else:
        score -= 1; details.append({"Factor": "Trend", "Status": "Down ↓", "Impact": "-1", "Color": "red"})

    # 2. Volume
    if last['RVOL'] > 2.0:
        score += 2; details.append({"Factor": "Volume", "Status": f"Spike ({last['RVOL']:.1f}x)", "Impact": "+2", "Color": "green"})
    else:
        details.append({"Factor": "Volume", "Status": "Normal", "Impact": "0", "Color": "grey"})

    # 3. RSI
    if last['RSI'] < 35:
        score += 2; details.append({"Factor": "RSI", "Status": "Oversold", "Impact": "+2", "Color": "green"})
    elif last['RSI'] > 70:
        score -= 2; details.append({"Factor": "RSI", "Status": "Overbought", "Impact": "-2", "Color": "red"})
    else:
        details.append({"Factor": "RSI", "Status": "Neutral", "Impact": "0", "Color": "grey"})

    # 4. Zone & Fibs
    price = last['Close']
    # Cek apakah harga di Golden Pocket (0.5 - 0.618)
    gp_high = fib_levels.get('0.5', 0)
    gp_low = fib_levels.get('0.618 (Golden)', 0)
    
    if gp_low <= price <= gp_high:
        score += 2; details.append({"Factor": "Fibonacci", "Status": "Golden Pocket!", "Impact": "+2", "Color": "green"})
    elif price < fib_levels.get('0.786', 0):
        score += 1; details.append({"Factor": "Fibonacci", "Status": "Deep Discount", "Impact": "+1", "Color": "green"})
    else:
        details.append({"Factor": "Fibonacci", "Status": "Extension/Premium", "Impact": "0", "Color": "grey"})

    # 5. Order Block
    for ob in obs:
        if ob['type'] == 'Bull' and (last['Low'] <= ob['h']):
            score += 3; details.append({"Factor": "Order Block", "Status": "Hit Bullish OB", "Impact": "+3", "Color": "green"})
        if ob['type'] == 'Bear' and (last['High'] >= ob['l']):
            score -= 3; details.append({"Factor": "Order Block", "Status": "Hit Bearish OB", "Impact": "-3", "Color": "red"})

    # Final Action
    action = "WAIT"
    if score >= 6: action = "STRONG BUY"
    elif score >= 4: action = "BUY"
    elif score <= -4: action = "SELL"
    elif score <= -6: action = "STRONG SELL"
    
    return score, action, details

# --- 5. MAIN APP ---

st.sidebar.title("⚔️ SniperPalmerah")
app_mode = st.sidebar.radio("Mode:", ["🎯 Deep Analysis", "🚀 Accumulation Scanner"], index=0)
source_opt = st.sidebar.selectbox("Exchange", ['binance', 'gateio', 'mexc', 'bybit', 'kucoin'], index=0)

# ==========================================
# MODE 1: DEEP ANALYSIS
# ==========================================
if app_mode == "🎯 Deep Analysis":
    symbols = get_symbols(source_opt)
    default_sym = symbols.index("BTC/USDT") if "BTC/USDT" in symbols else 0
    
    c1, c2 = st.sidebar.columns(2)
    sym = c1.selectbox("Pair", symbols, index=default_sym)
    tf = c2.selectbox("TF", ['1m', '5m', '15m', '1h', '4h', '1d'], index=4)
    
    sens = st.sidebar.slider("Structure Sensitivity", 2, 20, get_best_sensitivity(tf))
    
    df = fetch_data(sym, tf, ex_id=source_opt)
    
    if not df.empty:
        df = calculate_all(df)
        signals, struct = get_smc_structure(df, order=sens)
        obs = get_order_blocks(df)
        
        # NEW ANALYSIS CALLS
        fib_levels, swing_h, swing_l = analyze_fib(df)
        is_db, db_neck = analyze_double_bottom(df)
        is_hidden_div = analyze_hidden_divergence(df)
        reversal_candles = analyze_reversal_candles(df)
        
        score, action, details = detailed_analysis(df, struct, obs, fib_levels)
        last = df.iloc[-1]
        
        # --- HEADER ---
        st.title(f"📊 {sym}")
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Price", f"{last['Close']:,.4f}")
        m2.metric("Score", f"{score}")
        m3.metric("RSI", f"{last['RSI']:.1f}")
        m4.metric("Action", action)

        # --- RECOMMENDATION SECTION (SPOT) ---
        st.subheader("💡 Spot Recommendation")
        
        # Determine Entry, SL, TP
        entry_zone = last['Close']
        sl_price = struct['last_low'] if struct['last_low'] != 0 else last['Close'] - (last['ATR']*2)
        
        # Smart TP based on Fibs
        tp1 = fib_levels.get('0.236', entry_zone)
        tp2 = fib_levels.get('0.0 (Low)', entry_zone) # This is High if trend up
        
        # Logic for displaying recommendation
        rec_html = ""
        if action in ["STRONG BUY", "BUY"]:
            rec_html = f"""
            <div class="alert-buy">
                <h3>🟢 Rekomendasi: BELI (BUY)</h3>
                <p><b>Alasan:</b> Skor {score} menunjukkan momentum positif.</p>
                <p><b>Entry Zone:</b> Market (Sekarang) atau Limit di <b>{fib_levels.get('0.618 (Golden)', 0):.4f}</b> (Golden Pocket)</p>
                <p><b>Stop Loss (SL):</b> <b>{sl_price:.4f}</b> (Di bawah Low Structure)</p>
                <p><b>Take Profit 1 (TP1):</b> <b>{tp1:.4f}</b> (Fib 0.236)</p>
                <p><b>Take Profit 2 (TP2):</b> <b>{swing_h:.4f}</b> (High Terakhir)</p>
            </div>
            """
        elif action in ["STRONG SELL", "SELL"]:
            rec_html = f"""
            <div class="alert-sell">
                <h3>🔴 Rekomendasi: JUAL (SELL)</h3>
                <p><b>Alasan:</b> Indikator menunjukkan pelemahan/momentum turun.</p>
                <p><b>Action:</b> Ambil Profit atau Cut Loss.</p>
            </div>
            """
        else:
            rec_html = f"""
            <div class="alert-wait">
                <h3>⚪ TUNGGU (WAIT)</h3>
                <p>Konfirmasi belum kuat. Tunggu candle close atau signal struktur.</p>
                <p>Area diskon favorit: <b>{fib_levels.get('0.618 (Golden)', 0):.4f}</b></p>
            </div>
            """
        st.markdown(rec_html, unsafe_allow_html=True)
        
        # --- SPECIAL PATTERN DETECTION STATUS ---
        st.subheader("🔍 Pattern Detection")
        cols = st.columns(4)
        
        # 1. Double Bottom
        db_status = "✅ Terdeteksi" if is_db else "❌ Tidak Ada"
        db_color = "green" if is_db else "grey"
        cols[0].markdown(f"**Double Bottom**<br><span style='color:{db_color}'>{db_status}</span>", unsafe_allow_html=True)
        
        # 2. Hidden Div
        hd_status = "✅ Terdeteksi" if is_hidden_div else "❌ Tidak Ada"
        hd_color = "green" if is_hidden_div else "grey"
        cols[1].markdown(f"**Hidden Div Bull**<br><span style='color:{hd_color}'>{hd_status}</span>", unsafe_allow_html=True)
        
        # 3. Reversal Candle
        rc_text = ", ".join(reversal_candles) if reversal_candles else "❌ Tidak Ada"
        cols[2].markdown(f"**Reversal Candle**<br>{rc_text}", unsafe_allow_html=True)
        
        # 4. Fib Position
        fib_pos = "Discount" if last['Close'] < fib_levels.get('0.5', 0) else "Premium"
        cols[3].metric("Fib Zone", fib_pos)

        # --- ANALYSIS BREAKDOWN TABLE ---
        with st.expander("📋 Detail Analisa Lengkap"):
            table_html = "<table style='width:100%'><thead><tr><th>Factor</th><th>Status</th><th>Impact</th></tr></thead><tbody>"
            for d in details:
                table_html += f"<tr><td>{d['Factor']}</td><td>{d['Status']}</td><td style='color:{d['Color']}; font-weight:bold;'>{d['Impact']}</td></tr>"
            table_html += "</tbody></table>"
            st.markdown(table_html, unsafe_allow_html=True)

        # --- CHARTING ---
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)
        
        # Candles
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                                     name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
        
        # EMAs
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#FFB300', width=1), name='EMA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#42A5F5', width=1), name='EMA50'), row=1, col=1)
        
        # FIB LEVELS (Visual)
        for name, price in fib_levels.items():
            if name in ['0.5', '0.618 (Golden)', '0.786']:
                color = '#6200EA' if 'Golden' in name else '#B388FF'
                fig.add_hline(y=price, line_dash="dash", line_color=color, opacity=0.6, 
                              annotation_text=name, annotation_position="right", row=1, col=1)
        
        # Order Blocks
        for ob in obs:
            color = 'rgba(255, 235, 59, 0.3)' if ob['type'] == 'Bull' else 'rgba(156, 39, 176, 0.3)' 
            border_color = '#FBC02D' if ob['type'] == 'Bull' else '#8E24AA'
            fig.add_shape(type="rect", x0=ob['time'], y0=ob['l'], x1=df.index[-1], y1=ob['h'], 
                          line=dict(color=border_color, width=1, dash='dot'), fillcolor=color, row=1, col=1)
        
        # Pattern Markers
        # Double Bottom Marker
        if is_db:
            fig.add_trace(go.Scatter(x=[df.index[-1]], y=[df['Low'].iloc[-1]], mode='markers',
                                     marker=dict(symbol='star', size=15, color='gold'), name='Double Bottom'), row=1, col=1)
        
        # Reversal Candle Markers
        if reversal_candles:
            fig.add_trace(go.Scatter(x=[df.index[-1]], y=[last['Low']], mode='text', 
                                     text=['🔥'], textfont_size=20, name='Reversal'), row=1, col=1)

        # Volume
        colors_vol = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors_vol, opacity=0.4), row=2, col=1)

        fig.update_layout(template="plotly_white", height=800, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Data fetch failed.")

# ==========================================
# MODE 2: SCANNER
# ==========================================
else:
    st.title("🚀 Accumulation Scanner")
    st.info("Mencari coin Double Bottom + Hidden Div di area Diskon.")
    
    scan_tf = st.sidebar.selectbox("Scanner TF", ['1h', '4h', '1d'], index=1)
    limit = st.sidebar.slider("Top Pairs", 10, 100, 30)
    
    run_scan = st.sidebar.button("🔍 SCAN")
    placeholder = st.empty()
    
    if run_scan:
        with placeholder.container():
            progress = st.progress(0)
            status = st.empty()
            results = []
            
            ex = getattr(ccxt, source_opt)({'enableRateLimit': True})
            ex.load_markets()
            tickers = ex.fetch_tickers()
            pairs = sorted([t for s, t in tickers.items() if s.endswith('/USDT')], key=lambda x: x.get('quoteVolume', 0), reverse=True)[:limit]
            
            for i, t in enumerate(pairs):
                sym = t['symbol']
                progress.progress((i+1)/len(pairs))
                status.text(f"Scanning {sym}...")
                
                try:
                    df = fetch_data(sym, scan_tf, ex_id=source_opt)
                    if df.empty or len(df) < 50: continue
                    
                    df = calculate_all(df)
                    
                    # Logic Scanner
                    is_db, _ = analyze_double_bottom(df)
                    is_hd = analyze_hidden_divergence(df)
                    last = df.iloc[-1]
                    
                    # Filter: Minimal salah satu pattern terjadi + RSI Oversold/Normal
                    if (is_db or is_hd) and last['RSI'] < 60:
                        score = 0
                        if is_db: score += 3
                        if is_hd: score += 3
                        if last['RSI'] < 40: score += 1
                        
                        results.append({
                            'Pair': sym, 'Price': last['Close'], 'Score': score,
                            'Pattern': f"{'DB' if is_db else ''} {'HDiv' if is_hd else ''}".strip(),
                            'RSI': round(last['RSI'], 1)
                        })
                except: continue
            
            status.text("Done!")
            if results:
                st.dataframe(pd.DataFrame(results).sort_values('Score', ascending=False), use_container_width=True, hide_index=True)
            else:
                st.warning("No patterns found.")
