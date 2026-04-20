import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# ======================== KONFIGURASI ========================
st.set_page_config(
    page_title="Crypto Scanner - Ultimate Edition", 
    layout="wide",
    page_icon="🔮"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
<style>
    .big-font { font-size:20px !important; font-weight: bold; }
    .medium-font { font-size:16px !important; }
    .buy-box { background-color: #00ff0022; padding: 15px; border-radius: 10px; border-left: 5px solid #00ff00; }
    .wait-box { background-color: #ffaa0022; padding: 15px; border-radius: 10px; border-left: 5px solid #ffaa00; }
    .avoid-box { background-color: #ff000022; padding: 15px; border-radius: 10px; border-left: 5px solid #ff0000; }
    .whale-box { background-color: #aa00ff22; padding: 15px; border-radius: 10px; border-left: 5px solid #aa00ff; }
    .prediction-box { background-color: #00aaff22; padding: 15px; border-radius: 10px; border-left: 5px solid #00aaff; }
    .pump-badge { background-color: #ff4444; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .trend-up { color: #00ff00; font-weight: bold; }
    .trend-down { color: #ff4444; font-weight: bold; }
    .trend-sideways { color: #ffaa00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ======================== HEADER ========================
st.title("🔮 Crypto Scanner - Ultimate Edition")
st.markdown("""
<div class="medium-font">
<b>Fitur Lengkap:</b> Deteksi Whale + Prediksi Trend + Pola Chart Masa Depan + Garis Trend<br>
📌 <i>Dilengkapi prediksi arah harga 5-30 hari ke depan</i>
</div>
""", unsafe_allow_html=True)

# ======================== CACHED FUNCTIONS ========================
@st.cache_data(ttl=3600)
def get_all_usdt_pairs(exchange_name):
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        'timeout': 10000
    })
    markets = exchange.load_markets()
    usdt_pairs = [s for s in markets if s.endswith('/USDT') and markets[s]['spot']]
    
    valid_pairs = []
    progress_bar = st.progress(0)
    for i, sym in enumerate(usdt_pairs[:300]):
        try:
            ticker = exchange.fetch_ticker(sym)
            if ticker.get('quoteVolume', 0) > 10000:
                valid_pairs.append(sym)
        except:
            continue
        progress_bar.progress((i+1)/min(len(usdt_pairs), 300))
        time.sleep(0.05)
    
    progress_bar.empty()
    return valid_pairs

@st.cache_data(ttl=600)
def fetch_ohlcv_cached(symbol, exchange_name, timeframe='1d', limit=200):
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'},
        'timeout': 10000
    })
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if not ohlcv:
            return None
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ======================== INDIKATOR TEKNIKAL ========================
def calculate_indicators(df):
    if df is None or len(df) < 50:
        return None
    try:
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA50'] = df['close'].rolling(50).mean()
        df['MA200'] = df['close'].rolling(200).mean()
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp12 - exp26
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        df['Volume_MA20'] = df['volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA20'].replace(0, np.nan)
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        df['BB_Middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # ADX (Average Directional Index)
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        tr = true_range
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr_14)
        minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr_14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(14).mean()
        
        return df
    except Exception as e:
        print(f"Error indicators: {e}")
        return df

def calculate_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    if df is None or len(df) < senkou_b:
        return df
    try:
        df['tenkan'] = (df['high'].rolling(tenkan).max() + df['low'].rolling(tenkan).min()) / 2
        df['kijun'] = (df['high'].rolling(kijun).max() + df['low'].rolling(kijun).min()) / 2
        df['senkou_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(kijun)
        senkou_b_high = df['high'].rolling(senkou_b).max()
        senkou_b_low = df['low'].rolling(senkou_b).min()
        df['senkou_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(kijun)
        df['chikou'] = df['close'].shift(-kijun)
        df['future_senkou_a'] = ((df['tenkan'] + df['kijun']) / 2)
        df['future_senkou_b'] = ((df['high'].rolling(senkou_b).max() + df['low'].rolling(senkou_b).min()) / 2)
        return df
    except Exception as e:
        print(f"Error ichimoku: {e}")
        return df

def calculate_obv(df):
    if df is None or len(df) < 2:
        return df
    try:
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        
        if len(df) >= 20:
            obv_awal = df['OBV'].iloc[-20]
            obv_akhir = df['OBV'].iloc[-1]
            if obv_awal != 0:
                obv_change = (obv_akhir - obv_awal) / abs(obv_awal) * 100
                df['OBV_trend'] = obv_change > 5
                df['OBV_change_pct'] = obv_change
            else:
                df['OBV_trend'] = False
                df['OBV_change_pct'] = 0
        else:
            df['OBV_trend'] = False
            df['OBV_change_pct'] = 0
        return df
    except Exception as e:
        print(f"Error OBV: {e}")
        return df

# ======================== PREDIKSI TREND MASA DEPAN ========================
def predict_trend_direction(df, days_ahead=30):
    if df is None or len(df) < 50:
        return None
    
    try:
        current_price = df['close'].iloc[-1]
        predictions = []
        confidence_scores = []
        
        # METHOD 1: LINEAR REGRESSION
        x = np.arange(len(df)).reshape(-1, 1)
        y = df['close'].values
        
        model = LinearRegression()
        model.fit(x, y)
        
        future_x = np.arange(len(df), len(df) + days_ahead).reshape(-1, 1)
        linear_pred = model.predict(future_x)
        
        linear_slope = model.coef_[0]
        linear_direction = "up" if linear_slope > 0 else "down" if linear_slope < 0 else "sideways"
        linear_confidence = min(100, abs(linear_slope) / current_price * 100 * 10)
        
        predictions.append({
            'method': 'Linear Regression',
            'direction': linear_direction,
            'target_price': linear_pred[-1],
            'confidence': linear_confidence,
            'change_pct': (linear_pred[-1] - current_price) / current_price * 100
        })
        confidence_scores.append(linear_confidence)
        
        # METHOD 2: EMA CROSSOVER
        ema_20 = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        
        if ema_20 > ema_50:
            ema_direction = "up"
            ema_target = current_price * (1 + (ema_20 - ema_50) / ema_50 * 0.5)
            ema_confidence = 60
        elif ema_20 < ema_50:
            ema_direction = "down"
            ema_target = current_price * (1 - (ema_50 - ema_20) / ema_50 * 0.5)
            ema_confidence = 60
        else:
            ema_direction = "sideways"
            ema_target = current_price
            ema_confidence = 40
        
        predictions.append({
            'method': 'EMA Crossover',
            'direction': ema_direction,
            'target_price': ema_target,
            'confidence': ema_confidence,
            'change_pct': (ema_target - current_price) / current_price * 100
        })
        confidence_scores.append(ema_confidence)
        
        # METHOD 3: MACD MOMENTUM
        macd_hist = df['MACD_Hist'].values
        macd_trend = macd_hist[-5:].mean() - macd_hist[-20:-5].mean() if len(macd_hist) > 20 else 0
        
        if macd_trend > 0:
            macd_direction = "up"
            macd_target = current_price * (1 + abs(macd_trend) / current_price * 5)
            macd_confidence = 70 if macd_trend > 0.01 else 50
        else:
            macd_direction = "down"
            macd_target = current_price * (1 - abs(macd_trend) / current_price * 5)
            macd_confidence = 70 if macd_trend < -0.01 else 50
        
        predictions.append({
            'method': 'MACD Momentum',
            'direction': macd_direction,
            'target_price': macd_target,
            'confidence': macd_confidence,
            'change_pct': (macd_target - current_price) / current_price * 100
        })
        confidence_scores.append(macd_confidence)
        
        # METHOD 4: RSI MEAN REVERSION
        rsi_current = df['RSI'].iloc[-1]
        
        if rsi_current < 30:
            rsi_direction = "up"
            rsi_target = current_price * 1.05
            rsi_confidence = 80
        elif rsi_current > 70:
            rsi_direction = "down"
            rsi_target = current_price * 0.95
            rsi_confidence = 80
        elif rsi_current < 50:
            rsi_direction = "up_slow"
            rsi_target = current_price * 1.02
            rsi_confidence = 50
        else:
            rsi_direction = "down_slow"
            rsi_target = current_price * 0.98
            rsi_confidence = 50
        
        predictions.append({
            'method': 'RSI Mean Reversion',
            'direction': rsi_direction,
            'target_price': rsi_target,
            'confidence': rsi_confidence,
            'change_pct': (rsi_target - current_price) / current_price * 100
        })
        confidence_scores.append(rsi_confidence)
        
        # METHOD 5: ICHIMOKU CLOUD
        if 'senkou_a' in df.columns and 'senkou_b' in df.columns:
            last_senkou_a = df['senkou_a'].iloc[-1]
            last_senkou_b = df['senkou_b'].iloc[-1]
            future_senkou_a = df['future_senkou_a'].iloc[-1] if 'future_senkou_a' in df.columns else last_senkou_a
            future_senkou_b = df['future_senkou_b'].iloc[-1] if 'future_senkou_b' in df.columns else last_senkou_b
            
            if current_price > last_senkou_a and current_price > last_senkou_b:
                ichi_direction = "up"
                ichi_target = max(future_senkou_a, future_senkou_b) * 1.05
                ichi_confidence = 75
            elif current_price < last_senkou_a and current_price < last_senkou_b:
                ichi_direction = "down"
                ichi_target = min(future_senkou_a, future_senkou_b) * 0.95
                ichi_confidence = 65
            else:
                ichi_direction = "sideways"
                ichi_target = current_price
                ichi_confidence = 40
            
            predictions.append({
                'method': 'Ichimoku Cloud',
                'direction': ichi_direction,
                'target_price': ichi_target,
                'confidence': ichi_confidence,
                'change_pct': (ichi_target - current_price) / current_price * 100
            })
            confidence_scores.append(ichi_confidence)
        
        # KESIMPULAN PREDIKSI
        up_votes = sum(1 for p in predictions if p['direction'] in ['up', 'up_slow'])
        down_votes = sum(1 for p in predictions if p['direction'] in ['down', 'down_slow'])
        
        if up_votes > down_votes:
            final_direction = "BULLISH (Naik)"
            final_direction_icon = "📈"
        elif down_votes > up_votes:
            final_direction = "BEARISH (Turun)"
            final_direction_icon = "📉"
        else:
            final_direction = "SIDEWAYS (Mendatar)"
            final_direction_icon = "➡️"
        
        total_weight = sum(p['confidence'] for p in predictions)
        if total_weight > 0:
            weighted_target = sum(p['target_price'] * p['confidence'] for p in predictions) / total_weight
        else:
            weighted_target = current_price
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        if final_direction == "BULLISH (Naik)":
            estimated_timeframe = "1-2 minggu" if avg_confidence > 70 else "2-4 minggu"
        elif final_direction == "BEARISH (Turun)":
            estimated_timeframe = "1-2 minggu" if avg_confidence > 70 else "2-4 minggu"
        else:
            estimated_timeframe = "4-6 minggu (breakout belum jelas)"
        
        return {
            'final_direction': final_direction,
            'final_direction_icon': final_direction_icon,
            'weighted_target': round(weighted_target, 8),
            'avg_confidence': round(avg_confidence, 1),
            'estimated_timeframe': estimated_timeframe,
            'predicted_change_pct': round((weighted_target - current_price) / current_price * 100, 1),
            'detailed_predictions': predictions
        }
        
    except Exception as e:
        print(f"Error predict trend: {e}")
        return None

def detect_trendline(df, lookback=50):
    if df is None or len(df) < lookback:
        return None
    
    try:
        recent = df.tail(lookback)
        highs = recent['high'].values
        lows = recent['low'].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append((i, highs[i]))
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append((i, lows[i]))
        
        trendline_up = None
        trendline_down = None
        
        if len(swing_lows) >= 2:
            recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows
            if len(recent_lows) >= 2:
                x_lows = [p[0] for p in recent_lows]
                y_lows = [p[1] for p in recent_lows]
                slope_up = (y_lows[-1] - y_lows[0]) / (x_lows[-1] - x_lows[0]) if x_lows[-1] != x_lows[0] else 0
                intercept_up = y_lows[0] - slope_up * x_lows[0]
                
                trendline_up = {
                    'slope': slope_up,
                    'intercept': intercept_up,
                    'direction': 'up' if slope_up > 0 else 'down',
                    'current_value': slope_up * (len(highs)-1) + intercept_up
                }
        
        if len(swing_highs) >= 2:
            recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs
            if len(recent_highs) >= 2:
                x_highs = [p[0] for p in recent_highs]
                y_highs = [p[1] for p in recent_highs]
                slope_down = (y_highs[-1] - y_highs[0]) / (x_highs[-1] - x_highs[0]) if x_highs[-1] != x_highs[0] else 0
                intercept_down = y_highs[0] - slope_down * x_highs[0]
                
                trendline_down = {
                    'slope': slope_down,
                    'intercept': intercept_down,
                    'direction': 'up' if slope_down > 0 else 'down',
                    'current_value': slope_down * (len(highs)-1) + intercept_down
                }
        
        current_price = df['close'].iloc[-1]
        breakout_prediction = None
        
        if trendline_up and trendline_up['direction'] == 'up':
            distance_to_trendline = abs(current_price - trendline_up['current_value']) / current_price * 100
            if distance_to_trendline < 2:
                if current_price > trendline_up['current_value']:
                    breakout_prediction = "Harga DI ATAS garis trend naik -> support dinamis, potensi lanjut naik"
                else:
                    breakout_prediction = "Harga MENDEXATI garis trend naik -> potensi rebound dalam 2-5 hari"
        
        if trendline_down and trendline_down['direction'] == 'down':
            distance_to_resistance = abs(trendline_down['current_value'] - current_price) / current_price * 100
            if distance_to_resistance < 2:
                if current_price > trendline_down['current_value']:
                    breakout_prediction = "Harga BREAKOUT garis trend turun -> bullish signal!"
                else:
                    breakout_prediction = "Harga di BAWAH garis trend turun -> resistance dinamis, butuh volume besar buat tembus"
        
        return {
            'trendline_up': trendline_up,
            'trendline_down': trendline_down,
            'breakout_prediction': breakout_prediction,
            'swing_highs_count': len(swing_highs),
            'swing_lows_count': len(swing_lows)
        }
        
    except Exception as e:
        print(f"Error detect trendline: {e}")
        return None

def predict_breakout_time(df, current_price, nearest_resistance, nearest_support):
    if df is None or len(df) < 30:
        return None
    
    try:
        price_movement = df['close'].diff().abs().mean()
        distance_to_resistance = (nearest_resistance - current_price) / current_price * 100 if nearest_resistance > current_price else 0
        distance_to_support = (current_price - nearest_support) / current_price * 100 if current_price > nearest_support else 0
        
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else price_movement
        
        if distance_to_resistance > 0 and atr > 0:
            days_to_resistance = max(1, int(distance_to_resistance / (atr / current_price * 100)))
        else:
            days_to_resistance = 999
        
        vol_trend = df['Volume_Ratio'].tail(5).mean()
        
        if vol_trend > 1.5 and distance_to_resistance < 5:
            breakout_soon = True
            estimated_days = min(3, days_to_resistance)
            breakout_type = "UP (Resistance)"
            breakout_price = nearest_resistance
        elif distance_to_resistance < 3:
            breakout_soon = True
            estimated_days = days_to_resistance
            breakout_type = "UP (Resistance)"
            breakout_price = nearest_resistance
        elif distance_to_support < 3:
            breakout_soon = True
            estimated_days = 1
            breakout_type = "DOWN (Support)"
            breakout_price = nearest_support
        else:
            breakout_soon = False
            estimated_days = days_to_resistance if days_to_resistance < 999 else 7
            breakout_type = "Unknown"
            breakout_price = None
        
        return {
            'breakout_soon': breakout_soon,
            'estimated_days': estimated_days,
            'breakout_type': breakout_type,
            'breakout_price': breakout_price,
            'distance_to_resistance_pct': round(distance_to_resistance, 2),
            'distance_to_support_pct': round(distance_to_support, 2),
            'volume_trend': round(vol_trend, 1)
        }
        
    except Exception as e:
        print(f"Error predict breakout: {e}")
        return None

def predict_candlestick_pattern(df, days_ahead=5):
    if df is None or len(df) < 30:
        return None
    
    try:
        current_price = df['close'].iloc[-1]
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        volume_ratio = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else 1
        
        predictions = []
        
        if rsi < 30:
            predictions.append(f"Hari 1-2: Potensi HAMMER (reversal bullish) karena RSI oversold {rsi:.0f}")
        elif rsi > 70:
            predictions.append(f"Hari 1-2: Potensi SHOOTING STAR (reversal bearish) karena RSI overbought {rsi:.0f}")
        
        if volume_ratio > 2:
            predictions.append(f"Hari 1: Potensi VOLUME SPIKE + MARUBOZU (pergerakan besar {'naik' if rsi < 60 else 'turun'})")
        
        bb_width = df['BB_Width'].iloc[-1] if 'BB_Width' in df.columns else 0.1
        if bb_width < 0.05:
            predictions.append(f"Hari 2-3: Potensi LONG LEGGED DOJI (volatility expansion) karena squeeze {bb_width*100:.1f}%")
        
        macd_hist = df['MACD_Hist'].iloc[-3:].values if 'MACD_Hist' in df.columns else [0,0,0]
        if len(macd_hist) >= 3 and macd_hist[-1] > macd_hist[-2] > macd_hist[-3]:
            predictions.append("Hari 3-5: Cenderung GREEN CANDLE berturut-turut (momentum bullish)")
        elif len(macd_hist) >= 3 and macd_hist[-1] < macd_hist[-2] < macd_hist[-3]:
            predictions.append("Hari 3-5: Cenderung RED CANDLE berturut-turut (momentum bearish)")
        else:
            predictions.append("Hari 3-5: Cenderung DOJI atau SPINNING TOP (konsolidasi)")
        
        return predictions
        
    except Exception as e:
        print(f"Error predict candlestick: {e}")
        return ["Tidak bisa memprediksi pola candlestick"]

def monte_carlo_simulation(df, days_ahead=30, simulations=500):
    if df is None or len(df) < 50:
        return None
    
    try:
        returns = df['close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        current_price = df['close'].iloc[-1]
        
        simulated_prices = []
        final_prices = []
        
        for _ in range(simulations):
            prices = [current_price]
            for _ in range(days_ahead):
                random_return = np.random.normal(mean_return, std_return)
                prices.append(prices[-1] * (1 + random_return))
            simulated_prices.append(prices)
            final_prices.append(prices[-1])
        
        final_prices = np.array(final_prices)
        percentile_10 = np.percentile(final_prices, 10)
        percentile_25 = np.percentile(final_prices, 25)
        percentile_50 = np.percentile(final_prices, 50)
        percentile_75 = np.percentile(final_prices, 75)
        percentile_90 = np.percentile(final_prices, 90)
        
        return {
            'current_price': current_price,
            'days_ahead': days_ahead,
            'p10': round(percentile_10, 8),
            'p25': round(percentile_25, 8),
            'p50': round(percentile_50, 8),
            'p75': round(percentile_75, 8),
            'p90': round(percentile_90, 8),
            'mean': round(final_prices.mean(), 8),
            'std': round(final_prices.std(), 8),
            'simulated_paths': simulated_prices[:5]
        }
        
    except Exception as e:
        print(f"Error monte carlo: {e}")
        return None

# ======================== WHALE DETECTION ========================
def detect_obv_divergence(df):
    if df is None or len(df) < 30:
        return False, "neutral", "Data tidak cukup"
    try:
        obv_values = df['OBV'].tail(20).values if 'OBV' in df.columns else np.zeros(20)
        price_values = df['close'].tail(20).values
        
        obv_slope = stats.linregress(range(len(obv_values)), obv_values).slope if len(obv_values) > 1 else 0
        price_slope = stats.linregress(range(len(price_values)), price_values).slope if len(price_values) > 1 else 0
        
        if obv_slope > 0 and price_slope < 0:
            return True, "bullish", f"OBV naik {obv_slope:.0f} tapi harga turun {abs(price_slope):.0f} -> WHALE BELI DIAM-DIAM!"
        elif obv_slope < 0 and price_slope > 0:
            return True, "bearish", f"OBV turun {abs(obv_slope):.0f} tapi harga naik {price_slope:.0f} -> WHALE JUAL DIAM-DIAM!"
        return False, "neutral", "Tidak ada divergensi"
    except:
        return False, "neutral", "Error"

def detect_wick_manipulation(df):
    if df is None or len(df) < 5:
        return [], 0
    
    manipulations = []
    score = 0
    current_price = df['close'].iloc[-1]
    
    for i in range(1, 6):
        candle = df.iloc[-i]
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        lower_wick = min(candle['close'], candle['open']) - candle['low']
        candle_range = candle['high'] - candle['low']
        
        if candle_range > 0 and body > 0:
            wick_ratio = (upper_wick + lower_wick) / candle_range
            
            if wick_ratio > 0.6:
                if upper_wick > lower_wick * 2:
                    wick_to_price_ratio = upper_wick / current_price * 100
                    manipulations.append(f"Candle {i}: Wick ATAS ({upper_wick/body:.1f}x body, {wick_to_price_ratio:.1f}% dari harga) -> whale jual")
                    score += 2
                elif lower_wick > upper_wick * 2:
                    wick_to_price_ratio = lower_wick / current_price * 100
                    manipulations.append(f"Candle {i}: Wick BAWAH ({lower_wick/body:.1f}x body, {wick_to_price_ratio:.1f}% dari harga) -> whale beli")
                    score += 2
    
    return manipulations, score

def detect_volume_profile_whale(df):
    if df is None or len(df) < 30:
        return False, 0, ""
    try:
        recent = df.tail(50)
        price_bins = pd.cut(recent['close'], bins=10)
        vol_by_price = recent['volume'].groupby(price_bins).sum()
        
        if len(vol_by_price) > 0:
            max_vol = vol_by_price.max()
            total_vol = vol_by_price.sum()
            concentration = max_vol / total_vol if total_vol > 0 else 0
            
            if concentration > 0.3:
                max_bin = vol_by_price.idxmax()
                return True, round(concentration * 100, 1), f"Volume terkonsentrasi {concentration*100:.1f}% di range {max_bin}"
        return False, 0, ""
    except:
        return False, 0, ""

def detect_large_transactions(df):
    if df is None or len(df) < 10:
        return [], 0
    large_tx = []
    avg_vol = df['volume'].tail(30).mean()
    for i in range(1, 6):
        candle_vol = df['volume'].iloc[-i]
        if avg_vol > 0 and candle_vol > avg_vol * 3:
            large_tx.append(f"Candle {i}: {candle_vol/avg_vol:.1f}x normal")
    return large_tx, len(large_tx)

def detect_fakeout(df, current_price, nearest_resistance):
    if df is None or len(df) < 10:
        return False, 0, []
    fake_signals = []
    fake_score = 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    if current_price > nearest_resistance:
        vol_ratio = last['volume'] / df['volume'].tail(10).mean()
        if vol_ratio < 1.2:
            fake_signals.append(f"Breakout volume RENDAH ({vol_ratio:.1f}x) -> FAKEOUT!")
            fake_score += 2
    
    upper_wick = last['high'] - max(last['close'], last['open'])
    body = abs(last['close'] - last['open'])
    if body > 0 and upper_wick > body * 2 and current_price > nearest_resistance:
        fake_signals.append(f"Wick panjang {upper_wick/body:.1f}x body di resistance -> whale jual")
        fake_score += 2
    
    if prev['high'] > nearest_resistance and prev['close'] < nearest_resistance:
        fake_signals.append("Candle sebelumnya tembus tapi nutup di bawah -> FALSE BREAKOUT")
        fake_score += 1
    
    return fake_score >= 2, fake_score, fake_signals

# ======================== FUNGSI DETEKSI LAINNYA ========================
def detect_accumulation(df):
    if df is None or len(df) < 50:
        return False, "Data tidak cukup"
    try:
        recent = df.tail(30)
        price_range = (recent['high'].max() - recent['low'].min()) / recent['low'].min()
        is_sideways = price_range < 0.15
        avg_vol_ratio = recent['Volume_Ratio'].mean()
        is_volume_low = avg_vol_ratio < 0.8
        obv_trend = df['OBV_trend'].iloc[-1] if 'OBV_trend' in df else False
        
        if is_sideways and is_volume_low and obv_trend:
            return True, f"✅ Akumulasi: sideways {price_range:.1%}, volume turun {avg_vol_ratio:.2f}x, OBV naik {df['OBV_change_pct'].iloc[-1]:.1f}%"
        elif is_sideways and is_volume_low:
            return True, f"⚠️ Akumulasi awal: sideways {price_range:.1%}, volume turun {avg_vol_ratio:.2f}x (OBV belum konfirmasi)"
        else:
            return False, f"Tidak akumulasi (range: {price_range:.1%}, volume: {avg_vol_ratio:.2f}x)"
    except:
        return False, "Error"

def detect_volume_spike(df, lookback=5, threshold=2.0):
    if df is None:
        return False, 0
    try:
        max_vol_ratio = df['Volume_Ratio'].tail(lookback).max()
        if max_vol_ratio > threshold:
            return True, round(max_vol_ratio, 1)
        return False, 0
    except:
        return False, 0

def detect_volatility_squeeze(df, period=20):
    if df is None or len(df) < period:
        return False, 0
    try:
        current_width = df['BB_Width'].iloc[-1]
        avg_width = df['BB_Width'].tail(period).mean()
        if avg_width > 0:
            squeeze_pct = (1 - current_width / avg_width) * 100
            is_squeeze = squeeze_pct > 50
            return is_squeeze, round(squeeze_pct, 1)
        return False, 0
    except:
        return False, 0

def detect_breakout(df, lookback=20):
    if df is None or len(df) < lookback:
        return False, 0
    try:
        recent = df.tail(lookback)
        resistance = recent['high'].max()
        current_price = df['close'].iloc[-1]
        vol_ratio = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
        if current_price > resistance and vol_ratio > 1.5:
            return True, round(resistance, 8)
        return False, 0
    except:
        return False, 0

def detect_support_resistance(df):
    if df is None or len(df) < 20:
        return None, None
    try:
        current_price = df['close'].iloc[-1]
        lows = df['low'].tail(50)
        supports = [low for low in lows if low < current_price]
        nearest_support = max(supports) if supports else current_price * 0.95
        highs = df['high'].tail(50)
        resistances = [high for high in highs if high > current_price]
        nearest_resistance = min(resistances) if resistances else current_price * 1.05
        return round(nearest_support, 8), round(nearest_resistance, 8)
    except:
        return None, None

def get_entry_stop_loss(df, current_price, nearest_support):
    if df is None or len(df) < 20:
        return current_price * 0.95, current_price * 0.90
    atr = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else current_price * 0.02
    conservative_entry = nearest_support if nearest_support and nearest_support < current_price else current_price * 0.97
    recent_lows = df['low'].tail(20)
    swing_low = recent_lows.min()
    stop_loss = swing_low * 0.98 if swing_low < conservative_entry else conservative_entry * 0.95
    if conservative_entry - stop_loss > conservative_entry * 0.15:
        stop_loss = conservative_entry * 0.85
    return round(conservative_entry, 8), round(stop_loss, 8)

def calculate_targets(entry_price, atr):
    tp1 = entry_price + (atr * 2)
    tp2 = entry_price + (atr * 3)
    tp3 = entry_price + (atr * 5)
    return round(tp1, 8), round(tp2, 8), round(tp3, 8)

# ======================== POLA CHART ========================
def detect_double_bottom(df):
    if df is None or len(df) < 40:
        return False
    try:
        lows = df['low'].tail(40)
        valley1_idx = lows.idxmin()
        valley2_candidates = lows.loc[valley1_idx + 10:]
        if valley2_candidates.empty:
            return False
        valley2_idx = valley2_candidates.idxmin()
        valley1 = df.loc[valley1_idx, 'low']
        valley2 = df.loc[valley2_idx, 'low']
        if abs(valley1 - valley2) / valley1 > 0.05:
            return False
        middle_high = df.loc[valley1_idx:valley2_idx, 'high'].max()
        current_price = df['close'].iloc[-1]
        return current_price > middle_high * 0.98
    except:
        return False

def detect_bullish_flag(df):
    if df is None or len(df) < 30:
        return False
    try:
        first_10 = df.iloc[-30:-20]
        pole_change = (first_10['close'].iloc[-1] - first_10['close'].iloc[0]) / first_10['close'].iloc[0]
        last_20 = df.tail(20)
        flag_range = (last_20['high'].max() - last_20['low'].min()) / last_20['low'].min()
        vol_ratio_mean = last_20['Volume_Ratio'].mean()
        return pole_change > 0.10 and flag_range < 0.08 and vol_ratio_mean < 0.8
    except:
        return False

def detect_cup_handle(df):
    if df is None or len(df) < 50:
        return False
    try:
        min_price = df['low'].tail(50).min()
        min_idx = df['low'].tail(50).idxmin()
        after_bottom = df.loc[min_idx:].tail(20)
        recovery = (after_bottom['close'].iloc[-1] - min_price) / min_price
        last_10 = df.tail(10)
        handle_range = (last_10['high'].max() - last_10['low'].min()) / last_10['low'].min()
        return recovery > 0.15 and handle_range < 0.05
    except:
        return False

def detect_head_shoulders(df):
    if df is None or len(df) < 60:
        return False, ""
    try:
        highs = df['high'].tail(60).values
        peaks = []
        for i in range(2, len(highs)-2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 3:
            recent_peaks = peaks[-3:]
            left_shoulder = recent_peaks[0][1]
            head = recent_peaks[1][1]
            right_shoulder = recent_peaks[2][1]
            
            if head > left_shoulder and head > right_shoulder:
                return True, "Head and Shoulders (bearish reversal) terdeteksi"
        
        return False, ""
    except:
        return False, ""

# ======================== FUNGSI UTAMA ANALISIS ========================
def analyze_coin_spot(symbol, exchange_name, btc_df=None):
    try:
        daily = fetch_ohlcv_cached(symbol, exchange_name, '1d', limit=200)
        if daily is None or len(daily) < 50:
            return None
        
        daily = calculate_indicators(daily)
        if daily is None:
            return None
        daily = calculate_ichimoku(daily)
        daily = calculate_obv(daily)
        
        last = daily.iloc[-1]
        current_price = last['close']
        
        # Deteksi berbagai kondisi
        is_accum, acc_reason = detect_accumulation(daily)
        has_volume_spike, spike_ratio = detect_volume_spike(daily)
        is_squeeze, squeeze_pct = detect_volatility_squeeze(daily)
        has_breakout, breakout_level = detect_breakout(daily)
        
        # Support & Resistance
        nearest_support, nearest_resistance = detect_support_resistance(daily)
        if nearest_support is None:
            nearest_support = current_price * 0.95
        if nearest_resistance is None:
            nearest_resistance = current_price * 1.05
        
        # Entry & Stop Loss
        conservative_entry, stop_loss = get_entry_stop_loss(daily, current_price, nearest_support)
        atr_value = last['ATR'] if not pd.isna(last['ATR']) else current_price * 0.02
        tp1, tp2, tp3 = calculate_targets(conservative_entry, atr_value)
        
        # Risk/Reward
        risk = conservative_entry - stop_loss
        reward = tp1 - conservative_entry
        rr_ratio = reward / risk if risk > 0 else 0
        
        # PREDIKSI MASA DEPAN
        trend_prediction = predict_trend_direction(daily, days_ahead=30)
        trendline_analysis = detect_trendline(daily, lookback=50)
        breakout_prediction = predict_breakout_time(daily, current_price, nearest_resistance, nearest_support)
        candlestick_prediction = predict_candlestick_pattern(daily, days_ahead=5)
        monte_carlo = monte_carlo_simulation(daily, days_ahead=30, simulations=500)
        
        # WHALE DETECTION
        has_divergence, div_type, div_reason = detect_obv_divergence(daily)
        wick_signals, wick_score = detect_wick_manipulation(daily)
        has_vol_concentration, vol_conc_pct, vol_profile_reason = detect_volume_profile_whale(daily)
        large_tx, tx_count = detect_large_transactions(daily)
        is_fakeout, fake_score, fake_signals = detect_fakeout(daily, current_price, nearest_resistance)
        
        # Whale confidence
        whale_confidence = 0
        whale_signals = []
        
        if has_divergence and div_type == "bullish":
            whale_confidence += 3
            whale_signals.append(f"🐋 {div_reason}")
        if wick_score >= 2:
            whale_confidence += 2
            for w in wick_signals[:2]:
                whale_signals.append(f"📊 {w}")
        if has_vol_concentration:
            whale_confidence += 2
            whale_signals.append(f"📊 {vol_profile_reason}")
        if tx_count >= 1:
            whale_confidence += 1
            for tx in large_tx[:1]:
                whale_signals.append(f"🔊 {tx}")
        if is_fakeout:
            whale_confidence -= 2
        
        # SCORING
        score = 0
        reasons = []
        
        if is_accum:
            score += 3
            reasons.append(acc_reason)
        
        if whale_confidence >= 5:
            score += 3
            reasons.append(f"🐋 WHALE TERKONFIRMASI (conf {whale_confidence}/7)")
        elif whale_confidence >= 3:
            score += 2
            reasons.append(f"🐋 Ada aktivitas whale (conf {whale_confidence}/7)")
        
        if last['close'] > last['MA200']:
            score += 2
            reasons.append("✅ Harga di atas MA200")
        elif last['close'] > last['MA50']:
            score += 1
            reasons.append("📈 Harga di atas MA50")
        
        above_cloud = last['close'] > last['senkou_a'] and last['close'] > last['senkou_b']
        if above_cloud:
            score += 2
            reasons.append("✅ Harga di atas Cloud")
        
        tk_cross = last['tenkan'] > last['kijun']
        if tk_cross:
            score += 1
            reasons.append("✅ TK Cross (golden cross)")
        
        if 40 <= last['RSI'] <= 60:
            score += 1
            reasons.append(f"📊 RSI {last['RSI']:.1f}")
        
        if last['MACD'] > last['Signal']:
            score += 1
            reasons.append("✅ MACD bullish")
        
        if has_volume_spike:
            score += 1
            reasons.append(f"🔊 Volume spike {spike_ratio}x!")
        
        if is_squeeze:
            score += 1.5
            reasons.append(f"🔥 Squeeze {squeeze_pct}%!")
        
        if has_breakout:
            score += 1.5
            reasons.append(f"🚀 Breakout!")
        
        if detect_double_bottom(daily):
            score += 1
            reasons.append("📉 Double Bottom")
        if detect_bullish_flag(daily):
            score += 1
            reasons.append("🚩 Bullish Flag")
        if detect_cup_handle(daily):
            score += 1
            reasons.append("🏆 Cup and Handle")
        
        hs_detected, hs_reason = detect_head_shoulders(daily)
        if hs_detected:
            score -= 1
            reasons.append(f"⚠️ {hs_reason}")
        
        if daily['OBV_trend'].iloc[-1]:
            score += 1
            reasons.append(f"🐋 OBV naik {daily['OBV_change_pct'].iloc[-1]:.1f}%")
        
        # STATUS
        if score >= 7:
            status = "🚀 STRONG BUY - SIAP PUMP!"
            confidence = "Tinggi"
            action = "BELI"
        elif score >= 5:
            status = "📈 BUY - POTENSI PUMP"
            confidence = "Sedang"
            action = "BELI"
        elif score >= 3:
            status = "⏳ ACCUMULATION - PANTAU"
            confidence = "Rendah"
            action = "PANTAU"
        else:
            status = "⏸️ HOLD / WAIT"
            confidence = "Sangat Rendah"
            action = "HOLD"
        
        # Potensi pump
        if trend_prediction:
            if trend_prediction['final_direction'] == "BULLISH (Naik)":
                pump_potential = f"{abs(trend_prediction['predicted_change_pct'])}% dalam {trend_prediction['estimated_timeframe']}"
            else:
                pump_potential = f"Turun {abs(trend_prediction['predicted_change_pct'])}% (bearish)"
        else:
            pump_potential = "Tidak terprediksi"
        
        # Entry timing
        entry_timing_status = "⏳ PANTAU"
        entry_timing_reason = ""
        
        if breakout_prediction and breakout_prediction['breakout_soon']:
            if breakout_prediction['breakout_type'] == "UP (Resistance)":
                entry_timing_status = "⚡ AGGRESSIVE ENTRY"
                entry_timing_reason = f"Prediksi breakout dalam {breakout_prediction['estimated_days']} hari ke resistance {breakout_prediction['breakout_price']:.6f}"
            elif breakout_prediction['breakout_type'] == "DOWN (Support)":
                entry_timing_status = "⬇️ TUNGGU PULLBACK"
                entry_timing_reason = f"Prediksi menyentuh support {breakout_prediction['breakout_price']:.6f} dalam {breakout_prediction['estimated_days']} hari"
        elif conservative_entry < current_price:
            entry_timing_status = "⬇️ TUNGGU PULLBACK"
            entry_timing_reason = f"Pasang limit order di {conservative_entry:.6f}"
        elif conservative_entry > current_price:
            entry_timing_status = "✅ BELI SEKARANG"
            entry_timing_reason = f"Harga {current_price:.6f} sudah di bawah entry"
        else:
            entry_timing_status = "⏳ PASANG LIMIT"
            entry_timing_reason = f"Pasang limit order di {conservative_entry:.6f}"
        
        tp1_pct = ((tp1 - conservative_entry) / conservative_entry) * 100
        tp2_pct = ((tp2 - conservative_entry) / conservative_entry) * 100
        tp3_pct = ((tp3 - conservative_entry) / conservative_entry) * 100
        
        beginner_summary = f"""
        **Gampangnya:** 
        - 📊 Skor {score:.1f}/10 -> {status}
        - 🔮 Prediksi trend: {trend_prediction['final_direction'] if trend_prediction else 'N/A'} ({trend_prediction['avg_confidence'] if trend_prediction else 0}% confidence)
        - ⏰ Timing: {entry_timing_status} - {entry_timing_reason}
        - 💰 Entry: {conservative_entry:.6f}
        - 🛑 Stop loss: {stop_loss:.6f}
        - 🎯 Target: +{tp1_pct:.1f}% -> +{tp2_pct:.1f}% -> +{tp3_pct:.1f}%
        """
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 8),
            'entry_price': conservative_entry,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'tp1_pct': round(tp1_pct, 1),
            'tp2_pct': round(tp2_pct, 1),
            'tp3_pct': round(tp3_pct, 1),
            'rr_ratio': round(rr_ratio, 2),
            'score': round(score, 1),
            'status': status,
            'action': action,
            'confidence': confidence,
            'pump_potential': pump_potential,
            'is_accum': is_accum,
            'has_volume_spike': has_volume_spike,
            'is_squeeze': is_squeeze,
            'has_breakout': has_breakout,
            'tk_cross': tk_cross,
            'above_cloud': above_cloud,
            'rsi': round(last['RSI'], 1),
            'volume_ratio': round(last['Volume_Ratio'], 2) if not pd.isna(last['Volume_Ratio']) else 0,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'reasons': reasons,
            'beginner_summary': beginner_summary,
            'daily_df': daily,
            'whale_confidence': whale_confidence,
            'whale_signals': whale_signals,
            'wick_signals': wick_signals,
            'trend_prediction': trend_prediction,
            'trendline_analysis': trendline_analysis,
            'breakout_prediction': breakout_prediction,
            'candlestick_prediction': candlestick_prediction,
            'monte_carlo': monte_carlo,
            'entry_timing_status': entry_timing_status,
            'entry_timing_reason': entry_timing_reason,
            'is_fakeout': is_fakeout,
            'has_divergence': has_divergence
        }
        
    except Exception as e:
        print(f"Error analyze {symbol}: {e}")
        return None

def quick_scan(symbol, exchange_name):
    try:
        daily = fetch_ohlcv_cached(symbol, exchange_name, '1d', limit=100)
        if daily is None or len(daily) < 50:
            return None
        daily = calculate_indicators(daily)
        daily = calculate_obv(daily)
        last = daily.iloc[-1]
        
        score = 0
        if daily['OBV_trend'].iloc[-1]:
            score += 2
        if last['close'] > last['MA200']:
            score += 1
        if last['close'] > last['MA50']:
            score += 1
        if last['Volume_Ratio'] < 0.7:
            score += 1
        if 40 <= last['RSI'] <= 60:
            score += 0.5
        
        return {
            'symbol': symbol,
            'price': round(last['close'], 8),
            'score': round(score, 1),
            'obv_trend': daily['OBV_trend'].iloc[-1]
        }
    except:
        return None

# ======================== PLOT CHART ========================
def plot_advanced_chart(df, symbol, support=None, resistance=None, trendline_data=None):
    if df is None or len(df) < 50:
        return go.Figure()
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{symbol} - Harga + Garis Trend", "Volume", "RSI"))
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Harga'
    ), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA50'], 
                            name='MA50', line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA200'], 
                            name='MA200', line=dict(color='red', width=1.5)), row=1, col=1)
    
    # Support & Resistance
    if support:
        fig.add_hline(y=support, line_dash="dash", line_color="green", 
                     annotation_text="Support", row=1, col=1)
    if resistance:
        fig.add_hline(y=resistance, line_dash="dot", line_color="red", 
                     annotation_text="Resistance", row=1, col=1)
    
    # Garis Trend
    if trendline_data:
        x_vals = np.arange(len(df))
        if trendline_data.get('trendline_up'):
            tu = trendline_data['trendline_up']
            trend_up_vals = tu['slope'] * x_vals + tu['intercept']
            fig.add_trace(go.Scatter(x=df['timestamp'], y=trend_up_vals,
                                    name='Trendline Up', line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
        
        if trendline_data.get('trendline_down'):
            td = trendline_data['trendline_down']
            trend_down_vals = td['slope'] * x_vals + td['intercept']
            fig.add_trace(go.Scatter(x=df['timestamp'], y=trend_down_vals,
                                    name='Trendline Down', line=dict(color='magenta', width=1, dash='dash')), row=1, col=1)
    
    # Volume
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=colors), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], name='RSI', line=dict(color='purple', width=1.5)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold", row=3, col=1)
    
    fig.update_layout(template="plotly_dark", height=800, hovermode='x unified')
    fig.update_xaxes(title_text="Tanggal", row=3, col=1)
    fig.update_yaxes(title_text="Harga", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

# ======================== STREAMLIT UI ========================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    
    exchange_name = st.selectbox(
        "Pilih Exchange", 
        ["binance", "bybit", "okx", "kucoin", "gate", "coinbase", "bitget", "kraken"], 
        index=0
    )
    
    scan_limit = st.slider("Jumlah coin yang di-scan", 50, 300, 150)
    min_score = st.slider("Minimal skor untuk ditampilkan", 0, 10, 3)
    auto_refresh = st.checkbox("Auto refresh setiap 30 menit", value=True)
    
    st.markdown("---")
    st.header("🔍 Manual Check")
    manual_symbol = st.text_input("Cek coin tertentu", placeholder="Contoh: BTC/USDT, ETH/USDT")
    manual_button = st.button("🔍 Analisis Manual", use_container_width=True)
    
    st.markdown("---")
    st.header("📖 Fitur Baru")
    st.markdown("""
    **🔮 Prediksi Trend** - Arah harga 5-30 hari
    **📊 Garis Trend** - Support/resistance dinamis
    **⚡ Prediksi Breakout** - Kapan dan di harga berapa
    **🎲 Monte Carlo** - Simulasi 500 kemungkinan
    **🕯️ Pola Candlestick** - Prediksi 5 hari ke depan
    """)

# Session state
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None
if 'last_scan' not in st.session_state:
    st.session_state.last_scan = None
if 'manual_result' not in st.session_state:
    st.session_state.manual_result = None

# Tombol scan
col1, col2 = st.columns([1, 3])
with col1:
    scan_button = st.button("🔍 MULAI SCAN", type="primary", use_container_width=True)

# Proses scan
if scan_button or (auto_refresh and st.session_state.scan_results is not None and 
                   (st.session_state.last_scan is None or 
                    (datetime.now() - st.session_state.last_scan).seconds > 1800)):
    
    with st.spinner("Mengambil data..."):
        pairs = get_all_usdt_pairs(exchange_name)
        if len(pairs) > scan_limit:
            pairs = pairs[:scan_limit]
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, sym in enumerate(pairs):
        status_text.text(f"Scanning {i+1}/{len(pairs)}: {sym}")
        res = quick_scan(sym, exchange_name)
        if res and res['score'] >= min_score:
            results.append(res)
        progress_bar.progress((i+1)/len(pairs))
        time.sleep(0.03)
    
    status_text.empty()
    progress_bar.empty()
    
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('score', ascending=False)
        st.session_state.scan_results = df_results
        st.session_state.last_scan = datetime.now()
        st.success(f"✅ Ditemukan {len(results)} coin")
        st.rerun()
    else:
        st.warning(f"Tidak ada coin dengan skor >= {min_score}")

# Tampilkan hasil scan
if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
    df = st.session_state.scan_results
    st.subheader(f"📊 Hasil Scan - {st.session_state.last_scan.strftime('%H:%M:%S') if st.session_state.last_scan else 'Sekarang'}")
    
    for _, row in df.head(10).iterrows():
        bg = "#00ff0015" if row['score'] >= 7 else "#44ff0015" if row['score'] >= 5 else "#ffaa0015"
        badge = "🚀 SIAP PUMP" if row['score'] >= 7 else "📈 POTENSI PUMP" if row['score'] >= 5 else "⏳ AKUMULASI"
        st.markdown(f"""
        <div style="background-color: {bg}; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
            <b>{row['symbol']}</b> | 💰 ${row['price']:.6f} | ⭐ Skor: {row['score']} | <span class="pump-badge">{badge}</span> | {'🐋 OBV Naik' if row['obv_trend'] else ''}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔍 Analisis Detail")
    selected_coin = st.selectbox("Pilih coin", df['symbol'].tolist())
    
    if selected_coin:
        with st.spinner(f"Menganalisis {selected_coin}..."):
            btc_df = fetch_ohlcv_cached('BTC/USDT', exchange_name, '1d', limit=100)
            if btc_df is not None:
                btc_df = calculate_indicators(btc_df)
            detail = analyze_coin_spot(selected_coin, exchange_name, btc_df)
            
            if detail:
                # Status box
                if detail['score'] >= 7:
                    box_class = "buy-box"
                elif detail['score'] >= 5:
                    box_class = "buy-box"
                elif detail['score'] >= 3:
                    box_class = "wait-box"
                else:
                    box_class = "avoid-box"
                
                st.markdown(f"""
                <div class="{box_class}">
                    <h3>{detail['status']}</h3>
                    <p><b>Skor:</b> {detail['score']}/10 | <b>Konfidensi:</b> {detail['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # PREDIKSI TREND BOX
                if detail['trend_prediction']:
                    tp = detail['trend_prediction']
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h4>🔮 PREDIKSI TREND {tp['final_direction_icon']}</h4>
                        <p><b>Arah:</b> <span class="{'trend-up' if 'BULLISH' in tp['final_direction'] else 'trend-down' if 'BEARISH' in tp['final_direction'] else 'trend-sideways'}">{tp['final_direction']}</span></p>
                        <p><b>Target Harga 30 Hari:</b> ${tp['weighted_target']:.6f} ({tp['predicted_change_pct']:+.1f}%)</p>
                        <p><b>Confidence:</b> {tp['avg_confidence']}% | <b>Estimasi Waktu:</b> {tp['estimated_timeframe']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📊 Detail Prediksi per Metode"):
                        for p in tp['detailed_predictions']:
                            direction_icon = "📈" if "up" in p['direction'] else "📉" if "down" in p['direction'] else "➡️"
                            st.write(f"{direction_icon} **{p['method']}**: {p['direction']} ({p['change_pct']:+.1f}%), confidence {p['confidence']:.0f}%")
                
                # GARIS TREND & BREAKOUT
                col1, col2 = st.columns(2)
                with col1:
                    if detail['trendline_analysis']:
                        ta = detail['trendline_analysis']
                        st.subheader("📐 Analisis Garis Trend")
                        if ta['trendline_up']:
                            st.write(f"- Trendline Naik: slope {ta['trendline_up']['slope']:.2e}")
                        if ta['trendline_down']:
                            st.write(f"- Trendline Turun: slope {ta['trendline_down']['slope']:.2e}")
                        if ta['breakout_prediction']:
                            st.info(f"💡 {ta['breakout_prediction']}")
                
                with col2:
                    if detail['breakout_prediction']:
                        bp = detail['breakout_prediction']
                        st.subheader("⚡ Prediksi Breakout")
                        if bp['breakout_soon']:
                            st.warning(f"🚨 BREAKOUT SOON! {bp['estimated_days']} hari ke {bp['breakout_type']} di ${bp['breakout_price']:.6f}")
                        else:
                            st.write(f"- Jarak ke resistance: {bp['distance_to_resistance_pct']}%")
                            st.write(f"- Jarak ke support: {bp['distance_to_support_pct']}%")
                        st.write(f"- Volume trend: {bp['volume_trend']}x normal")
                
                # MONTE CARLO
                if detail['monte_carlo']:
                    mc = detail['monte_carlo']
                    st.subheader("🎲 Monte Carlo Simulation (500 skenario)")
                    st.write("**Estimasi Harga 30 Hari ke Depan:**")
                    
                    pct_p90 = (mc['p90'] / mc['current_price'] - 1) * 100
                    pct_p50 = (mc['p50'] / mc['current_price'] - 1) * 100
                    pct_p10 = (mc['p10'] / mc['current_price'] - 1) * 100
                    
                    st.write(f"- 📈 Optimis (P90): ${mc['p90']:.6f} ({pct_p90:+.1f}%)")
                    st.write(f"- 📊 Realistis (P50): ${mc['p50']:.6f} ({pct_p50:+.1f}%)")
                    st.write(f"- 📉 Pesimis (P10): ${mc['p10']:.6f} ({pct_p10:+.1f}%)")
                    st.write(f"- 🎯 Range P10-P90: ${mc['p10']:.6f} - ${mc['p90']:.6f}")
                
                # PREDIKSI CANDLESTICK
                if detail['candlestick_prediction']:
                    st.subheader("🕯️ Prediksi Pola Candlestick (5 Hari)")
                    for pred in detail['candlestick_prediction'][:3]:
                        st.write(f"- {pred}")
                
                # WHALE DETECTION
                if detail['whale_confidence'] >= 3:
                    st.markdown(f"""
                    <div class="whale-box">
                        <h4>🐋 WHALE DETECTION (Confidence {detail['whale_confidence']}/7)</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    for ws in detail['whale_signals'][:3]:
                        st.write(f"- {ws}")
                
                # ENTRY TIMING
                if detail['entry_timing_status'] == "✅ BELI SEKARANG":
                    st.success(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
                elif detail['entry_timing_status'] == "⬇️ TUNGGU PULLBACK":
                    st.warning(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
                elif detail['entry_timing_status'] == "⚡ AGGRESSIVE ENTRY":
                    st.info(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
                else:
                    st.info(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
                
                # Info harga
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 Harga Saat Ini", f"${detail['current_price']:.6f}")
                with col2:
                    st.metric("🎯 Entry Point", f"${detail['entry_price']:.6f}")
                with col3:
                    st.metric("🛑 Stop Loss", f"${detail['stop_loss']:.6f}")
                with col4:
                    st.metric("📊 Risk/Reward", f"1:{detail['rr_ratio']:.1f}")
                
                # Target profit
                st.write("**🎯 Target Profit:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Target 1: ${detail['tp1']:.6f} (+{detail['tp1_pct']:.1f}%)")
                with col2:
                    st.info(f"Target 2: ${detail['tp2']:.6f} (+{detail['tp2_pct']:.1f}%)")
                with col3:
                    st.success(f"Target 3: ${detail['tp3']:.6f} (+{detail['tp3_pct']:.1f}%)")
                
                st.info(detail['beginner_summary'])
                st.write(f"**📊 Support:** ${detail['nearest_support']:.6f} | **Resistance:** ${detail['nearest_resistance']:.6f}")
                
                # Signal badges
                st.write("**📡 Signal:**")
                sig_cols = st.columns(6)
                with sig_cols[0]:
                    st.success("🐋 Whale" if detail['whale_confidence'] >= 3 else "⏳ Normal")
                with sig_cols[1]:
                    st.warning("🔊 Volume Spike" if detail['has_volume_spike'] else "📊 Normal")
                with sig_cols[2]:
                    st.error("🔥 Squeeze" if detail['is_squeeze'] else "📈 Normal")
                with sig_cols[3]:
                    st.success("🚀 Breakout" if detail['has_breakout'] else "🔒 Normal")
                with sig_cols[4]:
                    st.success("📈 TK Cross" if detail['tk_cross'] else "⚠️ No")
                with sig_cols[5]:
                    st.success("🐋 Divergence" if detail['has_divergence'] else "📊 No")
                
                if detail['wick_signals']:
                    with st.expander("⚠️ Peringatan Wick Panjang"):
                        for w in detail['wick_signals']:
                            st.write(f"- {w}")
                
                with st.expander("📝 Detail Analisis"):
                    for reason in detail['reasons'][:15]:
                        st.write(f"- {reason}")
                
                # Chart
                st.plotly_chart(plot_advanced_chart(
                    detail['daily_df'], 
                    detail['symbol'],
                    detail['nearest_support'],
                    detail['nearest_resistance'],
                    detail['trendline_analysis']
                ), use_container_width=True)
                
                st.caption("⚠️ **Disclaimer:** Prediksi berdasarkan data historis dan probabilitas. Bukan jaminan keuntungan. Selalu gunakan manajemen risiko!")

# Manual analysis
if manual_button and manual_symbol:
    symbol = manual_symbol.strip().upper()
    if not symbol.endswith('/USDT'):
        symbol += '/USDT'
    
    with st.spinner(f"Menganalisis {symbol}..."):
        btc_df = fetch_ohlcv_cached('BTC/USDT', exchange_name, '1d', limit=100)
        if btc_df is not None:
            btc_df = calculate_indicators(btc_df)
        result = analyze_coin_spot(symbol, exchange_name, btc_df)
        
        if result:
            st.session_state.manual_result = result
            st.success(f"✅ Analisis {symbol} selesai!")
        else:
            st.error(f"❌ Gagal menganalisis {symbol}")
        st.rerun()

# Tampilkan manual result
if st.session_state.manual_result:
    detail = st.session_state.manual_result
    
    st.markdown("---")
    st.subheader(f"🔍 Hasil Manual Analysis: {detail['symbol']}")
    
    if detail['score'] >= 7:
        box_class = "buy-box"
    elif detail['score'] >= 5:
        box_class = "buy-box"
    elif detail['score'] >= 3:
        box_class = "wait-box"
    else:
        box_class = "avoid-box"
    
    st.markdown(f"""
    <div class="{box_class}">
        <h3>{detail['status']}</h3>
        <p><b>Skor:</b> {detail['score']}/10 | <b>Konfidensi:</b> {detail['confidence']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # PREDIKSI TREND
    if detail['trend_prediction']:
        tp = detail['trend_prediction']
        st.markdown(f"""
        <div class="prediction-box">
            <h4>🔮 PREDIKSI TREND {tp['final_direction_icon']}</h4>
            <p><b>Arah:</b> <span class="{'trend-up' if 'BULLISH' in tp['final_direction'] else 'trend-down' if 'BEARISH' in tp['final_direction'] else 'trend-sideways'}">{tp['final_direction']}</span></p>
            <p><b>Target 30 Hari:</b> ${tp['weighted_target']:.6f} ({tp['predicted_change_pct']:+.1f}%)</p>
            <p><b>Confidence:</b> {tp['avg_confidence']}% | <b>Estimasi:</b> {tp['estimated_timeframe']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # WHALE
    if detail['whale_confidence'] >= 3:
        st.markdown(f"""
        <div class="whale-box">
            <h4>🐋 WHALE DETECTION (Confidence {detail['whale_confidence']}/7)</h4>
        </div>
        """, unsafe_allow_html=True)
        for ws in detail['whale_signals'][:3]:
            st.write(f"- {ws}")
    
    # TIMING
    if detail['entry_timing_status'] == "✅ BELI SEKARANG":
        st.success(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
    elif detail['entry_timing_status'] == "⬇️ TUNGGU PULLBACK":
        st.warning(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
    else:
        st.info(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Harga Saat Ini", f"${detail['current_price']:.6f}")
    with col2:
        st.metric("🎯 Entry Point", f"${detail['entry_price']:.6f}")
    with col3:
        st.metric("🛑 Stop Loss", f"${detail['stop_loss']:.6f}")
    with col4:
        st.metric("📊 Risk/Reward", f"1:{detail['rr_ratio']:.1f}")
    
    st.write("**🎯 Target Profit:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"Target 1: ${detail['tp1']:.6f} (+{detail['tp1_pct']:.1f}%)")
    with col2:
        st.info(f"Target 2: ${detail['tp2']:.6f} (+{detail['tp2_pct']:.1f}%)")
    with col3:
        st.success(f"Target 3: ${detail['tp3']:.6f} (+{detail['tp3_pct']:.1f}%)")
    
    st.info(detail['beginner_summary'])
    st.write(f"**📊 Support:** ${detail['nearest_support']:.6f} | **Resistance:** ${detail['nearest_resistance']:.6f}")
    
    with st.expander("📝 Detail Analisis"):
        for reason in detail['reasons'][:15]:
            st.write(f"- {reason}")
    
    st.plotly_chart(plot_advanced_chart(
        detail['daily_df'], 
        detail['symbol'],
        detail['nearest_support'],
        detail['nearest_resistance'],
        detail['trendline_analysis']
    ), use_container_width=True)

# Footer
st.markdown("---")
st.caption("""
**🔮 FITUR ULTIMATE:**
- Prediksi Trend 30 Hari (5 metode: Linear Regression, EMA, MACD, RSI, Ichimoku)
- Deteksi Garis Trend Otomatis
- Prediksi Breakout (Waktu & Harga)
- Monte Carlo Simulation (500 skenario)
- Prediksi Pola Candlestick 5 Hari
- Whale Detection (OBV Divergence, Volume Profile, Wick Manipulation)
""")
