import streamlit as st
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import numpy as np
import time

# --- SETTINGS ---
st.set_page_config(layout="wide", page_title="Tradingview Terminal (Ultimate)")

# --- HELPER FUNCTIONS ---

def get_best_sensitivity(timeframe):
    preset = {'1m': 4, '5m': 5, '15m': 6, '1h': 8, '4h': 10, '6h': 12, '8h': 12, '1d': 15, '3d': 20, '1w': 20}
    return preset.get(timeframe, 10)

def get_best_sr_periods(timeframe):
    preset = {'1m': (5, 20), '5m': (5, 20), '15m': (7, 25), '1h': (10, 30), '4h': (12, 40), '6h': (12, 40), '8h': (15, 50), '1d': (15, 50), '3d': (20, 60)}
    return preset.get(timeframe, (10, 30))

# DATA FETCHING
@st.cache_data(ttl=3600)
def get_exchange_symbols(exchange_id):
    import ccxt
    try:
        exchange_class = getattr(ccxt, exchange_id)
        ex = exchange_class({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
        ex.load_markets()
        symbols = [symbol for symbol, market in ex.markets.items() 
                   if market.get('type') == 'spot' and market.get('quote') == 'USDT']
        return sorted(symbols)
    except Exception as e:
        st.error(f"Gagal ambil simbol: {e}")
        return ["BTC/USDT", "ETH/USDT"]

def fetch_yahoo_data(symbol, timeframe, period='1y'):
    import yfinance as yf
    valid_intervals = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '60m', '4h': '1h', '6h':'1h', '8h':'1h', '1d': '1d', '3d':'1d', '1w': '1wk'}
    yf_interval = valid_intervals.get(timeframe, '1d')
    if yf_interval in ['1m', '5m', '15m', '30m', '60m']: period = '1mo'
    try:
        df = yf.Ticker(symbol).history(period=period, interval=yf_interval)
        if df.empty: return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df
    except: return pd.DataFrame()

def fetch_ccxt_data(symbol, timeframe, limit=300, exchange_id='binance'):
    import ccxt
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({'enableRateLimit': True})
        if timeframe not in exchange.timeframes:
            if timeframe in ['6h', '8h']: timeframe = '4h' 
            elif timeframe == '3d': timeframe = '1d'
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except: return pd.DataFrame()

# TECHNICAL CALCULATIONS
def calculate_all(df):
    if df.empty: return df
    df = df.copy()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    
    try:
        bbands = ta.bbands(df['Close'], length=20)
        if bbands is not None:
            df['BBU'] = bbands.iloc[:, 2]
            df['BBL'] = bbands.iloc[:, 0]
    except: pass

    try:
        macd = ta.macd(df['Close'])
        if macd is not None:
            df['MACD'] = macd.iloc[:, 0]
            df['MACD_Signal'] = macd.iloc[:, 1]
            df['MACD_Hist'] = macd.iloc[:, 2]
    except: pass

    try:
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            df['Stoch_K'] = stoch.iloc[:, 0]
            df['Stoch_D'] = stoch.iloc[:, 1]
    except: pass

    try:
        df['CDL_HAMMER'] = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name="hammer").astype(float)
        df['CDL_ENGULFING'] = ta.cdl_pattern(df['Open'], df['High'], df['Low'], df['Close'], name="engulfing").astype(float)
    except:
        df['CDL_HAMMER'] = 0.0
        df['CDL_ENGULFING'] = 0.0
    return df

# --- SMART MONEY CONCEPTS (SMC) LOGIC ---

def get_structure_labels(df, order=10):
    signals = []
    try:
        highs = argrelextrema(df['High'].values, np.greater, order=order)[0]
        lows = argrelextrema(df['Low'].values, np.less, order=order)[0]
        
        points = []
        for i in highs: points.append({'time': df.index[i], 'price': df['High'].iloc[i], 'type': 'high'})
        for i in lows: points.append({'time': df.index[i], 'price': df['Low'].iloc[i], 'type': 'low'})
        points = sorted(points, key=lambda x: x['time'])
        
        if len(points) < 3: return []

        last_point = points[-1]
        prev_point = points[-2]
        is_uptrend = False
        if prev_point['type'] == 'low' and last_point['type'] == 'high':
             if last_point['price'] > prev_point['price']: is_uptrend = True
        
        last_close = df['Close'].iloc[-1]
        last_time = df.index[-1]

        if is_uptrend:
            last_swing_low = [p for p in points if p['type'] == 'low'][-1]
            if last_close < last_swing_low['price']:
                signals.append({'time': last_time, 'price': last_swing_low['price'], 'type': 'CHoCH Bear', 'text': 'CHoCH'})
            last_swing_high = [p for p in points if p['type'] == 'high'][-1]
            if last_close > last_swing_high['price']:
                signals.append({'time': last_time, 'price': last_swing_high['price'], 'type': 'BOS Bull', 'text': 'BOS'})
        else:
            last_swing_high = [p for p in points if p['type'] == 'high'][-1]
            if last_close > last_swing_high['price']:
                signals.append({'time': last_time, 'price': last_swing_high['price'], 'type': 'CHoCH Bull', 'text': 'CHoCH'})
            last_swing_low = [p for p in points if p['type'] == 'low'][-1]
            if last_close < last_swing_low['price']:
                signals.append({'time': last_time, 'price': last_swing_low['price'], 'type': 'BOS Bear', 'text': 'BOS'})
                
    except: pass
    return signals

def get_fvg(df):
    fvgs = []
    try:
        for i in range(2, len(df)):
            if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                fvgs.append({
                    'type': 'Bullish', 'start_time': df.index[i-2], 'end_time': df.index[i],
                    'high': df['Low'].iloc[i], 'low': df['High'].iloc[i-2]
                })
            elif df['High'].iloc[i] < df['Low'].iloc[i-2]:
                fvgs.append({
                    'type': 'Bearish', 'start_time': df.index[i-2], 'end_time': df.index[i],
                    'high': df['Low'].iloc[i-2], 'low': df['High'].iloc[i]
                })
    except: pass
    return fvgs

def get_order_blocks(df, lookback=20):
    """
    FIX: Filter OB berdasarkan posisi harga sekarang.
    - Bullish OB harus di BAWAH harga (Support).
    - Bearish OB harus di ATAS harga (Resistance).
    """
    obs = []
    try:
        last_close = df['Close'].iloc[-1]
        
        # 1. Bullish OB (Support) -> Harus di bawah harga sekarang
        troughs = argrelextrema(df['Low'].values, np.less, order=5)[0]
        if len(troughs) > 0:
            last_trough_idx = troughs[-1]
            search_area = df.iloc[max(0, last_trough_idx-lookback):last_trough_idx]
            bearish_candles = search_area[search_area['Close'] < search_area['Open']]
            
            if not bearish_candles.empty:
                ob_candle = bearish_candles.iloc[-1]
                # FILTER: High dari OB harus lebih rendah dari harga sekarang
                if ob_candle['High'] < last_close:
                    obs.append({
                        'type': 'Bullish OB', 
                        'time': bearish_candles.index[-1], 
                        'high': ob_candle['High'], 
                        'low': ob_candle['Low']
                    })

        # 2. Bearish OB (Resistance) -> Harus di atas harga sekarang
        peaks = argrelextrema(df['High'].values, np.greater, order=5)[0]
        if len(peaks) > 0:
            last_peak_idx = peaks[-1]
            search_area = df.iloc[max(0, last_peak_idx-lookback):last_peak_idx]
            bullish_candles = search_area[search_area['Close'] > search_area['Open']]
            
            if not bullish_candles.empty:
                ob_candle = bullish_candles.iloc[-1]
                # FILTER: Low dari OB harus lebih tinggi dari harga sekarang
                if ob_candle['Low'] > last_close:
                    obs.append({
                        'type': 'Bearish OB', 
                        'time': bullish_candles.index[-1], 
                        'high': ob_candle['High'], 
                        'low': ob_candle['Low']
                    })
                    
    except: pass
    return obs

def get_liquidity_sweeps(df, window=10):
    sweeps = []
    try:
        peaks = argrelextrema(df['High'].values, np.greater, order=window)[0]
        troughs = argrelextrema(df['Low'].values, np.less, order=window)[0]
        
        if len(peaks) >= 2:
            last_close = df['Close'].iloc[-1]
            last_high = df['High'].iloc[-1]
            prev_peak_price = df['High'].iloc[peaks[-1]]
            if last_high > prev_peak_price and last_close < prev_peak_price:
                sweeps.append({'time': df.index[-1], 'price': prev_peak_price, 'type': 'Bearish Sweep'})

        if len(troughs) >= 2:
            last_close = df['Close'].iloc[-1]
            last_low = df['Low'].iloc[-1]
            prev_trough_price = df['Low'].iloc[troughs[-1]]
            if last_low < prev_trough_price and last_close > prev_trough_price:
                sweeps.append({'time': df.index[-1], 'price': prev_trough_price, 'type': 'Bullish Sweep'})
    except: pass
    return sweeps

# --- STANDARD LOGIC ---

def get_divergence(df, window=5):
    if df.empty or 'RSI' not in df.columns: return df
    df = df.copy()
    rsi_clean = df['RSI'].dropna()
    if len(rsi_clean) < window * 2: return df

    price_peaks = argrelextrema(df['Close'].values, np.greater, order=window)[0]
    price_troughs = argrelextrema(df['Close'].values, np.less, order=window)[0]
    rsi_peaks = argrelextrema(rsi_clean.values, np.greater, order=window)[0]
    rsi_troughs = argrelextrema(rsi_clean.values, np.less, order=window)[0]
    
    df['divergence'] = np.nan
    
    if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
        curr_p = price_peaks[-1]; prev_p = price_peaks[-2]; curr_r = rsi_peaks[-1]
        if curr_r < len(df):
            if df['Close'].iloc[curr_p] > df['Close'].iloc[prev_p] and df['RSI'].iloc[curr_r] < df['RSI'].iloc[rsi_peaks[-2]]:
                df.loc[df.index[curr_p], 'divergence'] = 'RegBear'
            if df['Close'].iloc[curr_p] < df['Close'].iloc[prev_p] and df['RSI'].iloc[curr_r] > df['RSI'].iloc[rsi_peaks[-2]]:
                df.loc[df.index[curr_p], 'divergence'] = 'HidBear'

    if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
        curr_p = price_troughs[-1]; prev_p = price_troughs[-2]; curr_r = rsi_troughs[-1]
        if curr_r < len(df):
            if df['Close'].iloc[curr_p] < df['Close'].iloc[prev_p] and df['RSI'].iloc[curr_r] > df['RSI'].iloc[rsi_troughs[-2]]:
                df.loc[df.index[curr_p], 'divergence'] = 'RegBull'
            if df['Close'].iloc[curr_p] > df['Close'].iloc[prev_p] and df['RSI'].iloc[curr_r] < df['RSI'].iloc[rsi_troughs[-2]]:
                df.loc[df.index[curr_p], 'divergence'] = 'HidBull'
    return df

def get_sr_levels(df, minor_w=5, major_w=20):
    levels = []
    m_peaks = argrelextrema(df['High'].values, np.greater, order=major_w)[0]
    m_troughs = argrelextrema(df['Low'].values, np.less, order=major_w)[0]
    for i in m_peaks[-3:]: levels.append({'price': df['High'].iloc[i], 'type': 'Major Resistance'})
    for i in m_troughs[-3:]: levels.append({'price': df['Low'].iloc[i], 'type': 'Major Support'})
    
    n_peaks = argrelextrema(df['High'].values, np.greater, order=minor_w)[0]
    n_troughs = argrelextrema(df['Low'].values, np.less, order=minor_w)[0]
    for i in n_peaks[-3:]: levels.append({'price': df['High'].iloc[i], 'type': 'Minor Resistance'})
    for i in n_troughs[-3:]: levels.append({'price': df['Low'].iloc[i], 'type': 'Minor Support'})
    return levels

def get_fib_levels(df, back_candles=50):
    recent = df.iloc[-back_candles:]
    high_idx = recent['High'].idxmax()
    low_idx = recent['Low'].idxmin()
    price_high = recent['High'].max()
    price_low = recent['Low'].min()
    
    levels = {}
    diff = price_high - price_low
    
    if high_idx > low_idx:
        # Uptrend
        levels['Trend'] = 'Bullish'
        levels['0.0 (Low)'] = price_low
        levels['1.0 (High)'] = price_high
        levels['0.236'] = price_low + (diff * 0.236)
        levels['0.382'] = price_low + (diff * 0.382)
        levels['0.5'] = price_low + (diff * 0.5)
        levels['0.618'] = price_low + (diff * 0.618)
        levels['0.786'] = price_low + (diff * 0.786)
    else:
        # Downtrend
        levels['Trend'] = 'Bearish'
        levels['0.0 (High)'] = price_high
        levels['1.0 (Low)'] = price_low
        levels['0.236'] = price_high - (diff * 0.236)
        levels['0.382'] = price_high - (diff * 0.382)
        levels['0.5'] = price_high - (diff * 0.5)
        levels['0.618'] = price_high - (diff * 0.618)
        levels['0.786'] = price_high - (diff * 0.786)
        
    return levels

# --- SIDEBAR ---
st.sidebar.title("⚙️ AnggaPro Settings")
app_mode = st.sidebar.radio("Mode:", ["🎯 Analisis Satuan", "🔥 Market Scanner"], index=0)
source_opt = st.sidebar.selectbox("Sumber Data / Exchange", ['gateio', 'mexc', 'kucoin', 'okx', 'Yahoo Finance'], index=0)

# ===========================================
# MODE 1: ANALISIS SATUAN
# ===========================================
if app_mode == "🎯 Analisis Satuan":
    
    selected_symbol = ""
    if source_opt != 'Yahoo Finance':
        symbols_list = get_exchange_symbols(source_opt)
        default_idx = symbols_list.index("BTC/USDT") if "BTC/USDT" in symbols_list else 0
        selected_symbol = st.sidebar.selectbox("Pilih Coin (Spot)", symbols_list, index=default_idx)
        exchange_id = source_opt
        tf_opt = st.sidebar.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '4h', '6h', '8h', '1d', '3d', '1w'], index=4)
    else:
        selected_symbol = st.sidebar.text_input("Symbol Yahoo (Contoh: BTC-USD)", "BTC-USD").upper()
        exchange_id = None
        tf_opt = st.sidebar.selectbox("Timeframe", ['1m', '5m', '15m', '1h', '1d', '1w'], index=4)

    symbol_input = selected_symbol
    
    auto_sens = get_best_sensitivity(tf_opt)
    auto_minor, auto_major = get_best_sr_periods(tf_opt)
    
    sensitivity = st.sidebar.slider("Sensitivitas Struktur (Swing)", 3, 25, auto_sens)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Smart Money Concepts")
    show_fvg = st.sidebar.checkbox("Tampilkan FVG", value=True)
    show_ob = st.sidebar.checkbox("Tampilkan Order Blocks", value=True)
    show_sweep = st.sidebar.checkbox("Tampilkan Liquidity Sweep", value=True)
    show_bos = st.sidebar.checkbox("Tampilkan BOS / CHoCH", value=True)
    
    st.sidebar.markdown("---")
    sr_minor_w = st.sidebar.slider("Periode Minor S/R", 3, 30, auto_minor)
    sr_major_w = st.sidebar.slider("Periode Major S/R", 10, 100, auto_major)
    
    run_btn = st.sidebar.button("🚀 Analisis Sekarang")

    if run_btn:
        with st.spinner(f"Memuat data {symbol_input}..."):
            if source_opt == 'Yahoo Finance':
                df = fetch_yahoo_data(symbol_input, tf_opt)
            else:
                df = fetch_ccxt_data(symbol_input, tf_opt, exchange_id=exchange_id)
            
            if not df.empty:
                df = calculate_all(df)
                df = get_divergence(df, window=sensitivity)
                sr_levels = get_sr_levels(df, minor_w=sr_minor_w, major_w=sr_major_w)
                fib_levels = get_fib_levels(df)
                
                # Calculate SMC
                fvgs = get_fvg(df) if show_fvg else []
                obs = get_order_blocks(df) if show_ob else []
                sweeps = get_liquidity_sweeps(df) if show_sweep else []
                bos_choch = get_structure_labels(df, order=sensitivity) if show_bos else []
                
                last = df.iloc[-1]
                
                # DASHBOARD SUMMARY
                st.subheader(f"📊 Dashboard {symbol_input} ({tf_opt})")
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                trend_status = "Netral"
                if 'EMA_20' in df.columns and 'SMA_50' in df.columns:
                    if last['EMA_20'] > last['SMA_50']: trend_status = "🟢 Bullish"
                    else: trend_status = "🔴 Bearish"

                col_sum1.metric("Harga", f"{last['Close']:.4f}", f"RSI: {last['RSI']:.1f}")
                col_sum2.metric("Trend (EMA)", trend_status)
                col_sum3.metric("Volatility (BB)", "Normal" if 'BBU' not in df.columns else ("Breakout Atas" if last['Close'] > last['BBU'] else "Breakout Bawah"))
                
                # MAIN CHART SETUP
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.02, row_heights=[0.5, 0.15, 0.15, 0.2])

                # --- ROW 1: PRICE & OVERLAY ---
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
                
                # MA & BBands
                fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='blue', width=1), name='SMA 50'), row=1, col=1)
                
                if 'BBU' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['BBU'], line=dict(color='gray', dash='dash'), name='BB Upper', opacity=0.3), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['BBL'], line=dict(color='gray', dash='dash'), name='BB Lower', opacity=0.3, fill='tonexty'), row=1, col=1)

                # S/R Lines
                for lvl in sr_levels:
                    color = 'green' if 'Support' in lvl['type'] else 'red'
                    dash = 'solid' if 'Major' in lvl['type'] else 'dot'
                    fig.add_hline(y=lvl['price'], line_dash=dash, line_color=color, opacity=0.5, row=1, col=1)

                # Fib Lines
                for lvl_name, price in fib_levels.items():
                    if lvl_name != 'Trend':
                        fig.add_hline(y=price, line_dash="dash", line_color="purple", opacity=0.4, 
                                      annotation_text=lvl_name, annotation_position="right", row=1, col=1)

                # Divergence
                colors = {'RegBull': 'lime', 'HidBull': 'green', 'RegBear': 'red', 'HidBear': 'darkred'}
                for sig, col in colors.items():
                    d = df[df['divergence'] == sig]
                    if not d.empty:
                        fig.add_trace(go.Scatter(
                            x=d.index, y=d['Low'] if 'Bull' in sig else d['High'],
                            mode='markers+text', marker=dict(symbol='triangle-up' if 'Bull' in sig else 'triangle-down', size=10, color=col),
                            text=sig, textposition="bottom center" if 'Bull' in sig else "top center",
                            name=sig), row=1, col=1)
                
                # Candle Patterns
                if 'CDL_HAMMER' in df.columns:
                    patterns_map = {'CDL_HAMMER': 'Hammer', 'CDL_ENGULFING': 'Engulfing'}
                    for c_col, p_name in patterns_map.items():
                         if c_col in df.columns:
                            bull = df[df[c_col] == 100]
                            bear = df[df[c_col] == -100]
                            if not bull.empty:
                                fig.add_trace(go.Scatter(x=bull.index, y=bull['Low'], mode='text', text=p_name, textfont=dict(color='lime', size=10), textposition="bottom center"), row=1, col=1)
                            if not bear.empty:
                                fig.add_trace(go.Scatter(x=bear.index, y=bear['High'], mode='text', text=p_name, textfont=dict(color='red', size=10), textposition="top center"), row=1, col=1)

                # --- SMART MONEY VISUALIZATION ---
                
                # 1. FVG
                if show_fvg and fvgs:
                    for fvg in fvgs[-5:]:
                        color_fill = 'rgba(0, 255, 0, 0.15)' if fvg['type'] == 'Bullish' else 'rgba(255, 0, 0, 0.15)'
                        line_col = 'green' if fvg['type'] == 'Bullish' else 'red'
                        fig.add_shape(type="rect", x0=fvg['start_time'], y0=fvg['low'], x1=df.index[-1], y1=fvg['high'],
                            line=dict(color=line_col, width=1, dash='dot'), fillcolor=color_fill, row=1, col=1)

                # 2. Order Blocks (FIXED)
                if show_ob and obs:
                    for ob in obs:
                        color_fill = 'rgba(255, 215, 0, 0.2)' if 'Bullish' in ob['type'] else 'rgba(128, 0, 128, 0.2)'
                        line_col = 'gold' if 'Bullish' in ob['type'] else 'purple'
                        fig.add_shape(type="rect", x0=ob['time'], y0=ob['low'], x1=df.index[-1], y1=ob['high'],
                            line=dict(color=line_col, width=2), fillcolor=color_fill, row=1, col=1)

                # 3. Liquidity Sweeps
                if show_sweep and sweeps:
                    for sw in sweeps:
                        fig.add_trace(go.Scatter(x=[sw['time']], y=[sw['price']], mode='markers+text',
                            marker=dict(symbol='x', size=15, color='cyan' if 'Bullish' in sw['type'] else 'magenta'),
                            text="Sweep", textposition="bottom center" if 'Bullish' in sw['type'] else "top center",
                            name=sw['type']), row=1, col=1)
                
                # 4. BOS & CHoCH
                if show_bos and bos_choch:
                    for sig in bos_choch:
                        if 'BOS' in sig['type']:
                            fig.add_trace(go.Scatter(x=[sig['time']], y=[sig['price']], mode='text',
                                text=f"<b>{sig['text']}</b>", textfont=dict(color='black', size=12, family='Arial Black'),
                                textposition="bottom center" if 'Bull' in sig['type'] else "top center",
                                name=sig['type']), row=1, col=1)
                        else: # CHoCH
                            fig.add_trace(go.Scatter(x=[sig['time']], y=[sig['price']], mode='text',
                                text=f"<b>{sig['text']}</b>", textfont=dict(color='black', size=12, family='Arial Black'),
                                textposition="bottom center" if 'Bull' in sig['type'] else "top center",
                                name=sig['type']), row=1, col=1)

                # --- INDICATOR ROWS ---
                if 'MACD' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='blue'), name='MACD'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='orange'), name='Signal'), row=2, col=1)
                    colors_h = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
                    fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Hist', marker_color=colors_h), row=2, col=1)
                    fig.add_hline(y=0, line_color='gray', row=2, col=1)

                fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

                if 'Stoch_K' in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='blue'), name='Stoch K'), row=4, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='orange'), name='Stoch D'), row=4, col=1)

                fig.update_layout(xaxis_rangeslider_visible=False, height=900, legend=dict(orientation="h"))
                fig.update_yaxes(fixedrange=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # --- DETAIL ANALISIS LENGKAP ---
                with st.expander("📋 Detail Analisis Lengkap"):
                    # 1. SMART MONEY ANALYSIS
                    st.subheader("🧠 Analisis Smart Money (Inti & Saran)")
                    
                    if bos_choch:
                        for sig in bos_choch:
                            if 'BOS' in sig['type']:
                                st.success(f"**{sig['type']} Terdeteksi:** Trend masih kuat. Probabilitas lanjut tinggi.")
                            else:
                                st.warning(f"**{sig['type']} Terdeteksi:** Peringatan Pembalikan! Hati-hati melanjutkan trend sebelumnya.")
                    else:
                        st.info("Tidak ada sinyal BOS/CHoCH baru. Trend sedang konsolidasi atau netral.")

                    if obs:
                        for ob in obs:
                            if 'Bullish' in ob['type']:
                                st.markdown(f"🟡 **Bullish OB:** Area Akumulasi di `{ob['low']:.4f} - {ob['high']:.4f}`. \n\n*Saran:* Buy area.")
                            else:
                                st.markdown(f"🟣 **Bearish OB:** Area Distribusi di `{ob['low']:.4f} - {ob['high']:.4f}`. \n\n*Saran:* Sell area.")
                    else:
                         st.write("Tidak ada Order Block valid (mungkin harga sedang trending kuat).")

                    if fvgs:
                        last_fvg = fvgs[-1]
                        st.markdown(f"⚡ **FVG {last_fvg['type']}:** Gap harga di area `{last_fvg['low']:.4f} - {last_fvg['high']:.4f}`.")

                    if sweeps:
                         st.markdown(f"💧 **Liquidity Sweep:** Pembersihan Stop Loss di `{sweeps[0]['price']:.4f}`.")

                    st.markdown("---")

                    # 2. FIBONACCI RETRACEMENT
                    st.subheader("📏 Fibonacci Retracement (50 Candle Terakhir)")
                    
                    trend_dir = fib_levels.get('Trend', 'Netral')
                    st.caption(f"Trend Dasar: **{trend_dir}**")
                    
                    col_fib1, col_fib2, col_fib3 = st.columns(3)
                    display_fib = {k: v for k, v in fib_levels.items() if k != 'Trend'}
                    
                    with col_fib1:
                        st.metric("Level 0.0", f"{display_fib.get('0.0 (Low)', display_fib.get('0.0 (High)', 0)):.4f}")
                        st.metric("Level 0.236", f"{display_fib.get('0.236', 0):.4f}")
                        st.metric("Level 0.382", f"{display_fib.get('0.382', 0):.4f}")
                    
                    with col_fib2:
                        st.metric("Level 0.5", f"{display_fib.get('0.5', 0):.4f}")
                        st.metric("Level 0.618 (Golden)", f"{display_fib.get('0.618', 0):.4f}")
                    
                    with col_fib3:
                        st.metric("Level 0.786", f"{display_fib.get('0.786', 0):.4f}")
                        st.metric("Level 1.0", f"{display_fib.get('1.0 (High)', display_fib.get('1.0 (Low)', 0)):.4f}")

                    st.markdown("---")

                    # 3. SUPPORT & RESISTANCE
                    st.subheader("🚧 Support & Resistance Area")
                    
                    major_levels = [l for l in sr_levels if 'Major' in l['type']]
                    minor_levels = [l for l in sr_levels if 'Minor' in l['type']]
                    
                    tab_major, tab_minor = st.tabs(["🔴 Major Levels (Kuat)", "🟡 Minor Levels (Jangka Pendek)"])
                    
                    with tab_major:
                        if major_levels:
                            df_major = pd.DataFrame(major_levels)
                            df_major['price'] = df_major['price'].apply(lambda x: f"{x:.4f}")
                            df_major = df_major.rename(columns={'price': 'Harga', 'type': 'Tipe'})
                            st.dataframe(df_major, use_container_width=True, hide_index=True)
                        else:
                            st.write("Tidak ada Major Level terdeteksi.")
                            
                    with tab_minor:
                        if minor_levels:
                            df_minor = pd.DataFrame(minor_levels)
                            df_minor['price'] = df_minor['price'].apply(lambda x: f"{x:.4f}")
                            df_minor = df_minor.rename(columns={'price': 'Harga', 'type': 'Tipe'})
                            st.dataframe(df_minor, use_container_width=True, hide_index=True)
                        else:
                            st.write("Tidak ada Minor Level terdeteksi.")

            else:
                st.error("Gagal mengambil data. Cek koneksi atau symbol.")

# ===========================================
# MODE 2: MARKET SCANNER
# ===========================================
else:
    st.info("Scanner memindai coin berdasarkan parameter teknikal + SMC.")
    scan_tf = st.sidebar.selectbox("Timeframe Scan", ['1h', '4h', '6h', '1d'], index=1)
    limit_scan = st.sidebar.slider("Jumlah Top Pair", 10, 50, 20)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filter Scanner")
    filter_rsi = st.sidebar.checkbox("RSI Oversold/Overbought", value=True)
    filter_macd = st.sidebar.checkbox("MACD Cross", value=True)
    filter_div = st.sidebar.checkbox("Divergence Detected", value=True)
    filter_smc = st.sidebar.checkbox("Sweep / BOS Detected", value=True)

    if st.sidebar.button("🔍 Start Scanner"):
        import ccxt
        progress = st.progress(0)
        status = st.empty()
        results = []
        
        if source_opt == 'Yahoo Finance':
            st.warning("Scanner hanya untuk Crypto (CCXT).")
        else:
            try:
                ex = getattr(ccxt, source_opt)({'enableRateLimit': True})
                tickers = ex.fetch_tickers()
                pairs = [v for k, v in tickers.items() if '/USDT' in k]
                pairs = sorted(pairs, key=lambda x: x['quoteVolume'] if x['quoteVolume'] else 0, reverse=True)[:limit_scan]
                
                for i, t in enumerate(pairs):
                    sym = t['symbol']
                    status.text(f"Scanning {sym} ({i+1}/{len(pairs)})")
                    progress.progress((i+1)/len(pairs))
                    
                    try:
                        df = fetch_ccxt_data(sym, scan_tf, limit=100, exchange_id=source_opt)
                        if df.empty: continue
                        
                        df = calculate_all(df)
                        df = get_divergence(df, window=get_best_sensitivity(scan_tf))
                        
                        last = df.iloc[-1]
                        signals = []
                        
                        if filter_rsi:
                            if last['RSI'] < 30: signals.append("RSI Oversold")
                            elif last['RSI'] > 70: signals.append("RSI Overbought")
                        
                        if filter_macd and 'MACD' in df.columns:
                            prev = df.iloc[-2]
                            if prev['MACD'] < prev['MACD_Signal'] and last['MACD'] > last['MACD_Signal']:
                                signals.append("MACD Bull Cross")
                            elif prev['MACD'] > prev['MACD_Signal'] and last['MACD'] < last['MACD_Signal']:
                                signals.append("MACD Bear Cross")
                                
                        if filter_div:
                            divs = df[df['divergence'].notna()].tail(1)
                            if not divs.empty:
                                signals.append(f"Div: {divs.iloc[0]['divergence']}")
                        
                        if filter_smc:
                            sweeps = get_liquidity_sweeps(df)
                            if sweeps: signals.append(f"Sweep: {sweeps[0]['type']}")
                            
                            bos = get_structure_labels(df, order=get_best_sensitivity(scan_tf))
                            if bos: signals.append(f"Structure: {bos[0]['type']}")

                        if signals:
                            results.append({
                                'Symbol': sym, 'Price': last['Close'], 
                                'RSI': round(last['RSI'],1), 
                                'MACD': 'Trend' if last['MACD'] > last['MACD_Signal'] else 'Weak',
                                'Signals': ", ".join(signals)
                            })
                    except: continue
                
                status.text("Selesai!")
                if results:
                    st.dataframe(pd.DataFrame(results))
                else:
                    st.warning("Tidak ada coin memenuhi kriteria.")
            except Exception as e:

                st.error(f"Error: {e}")

