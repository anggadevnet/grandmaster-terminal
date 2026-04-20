import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
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
    .ichimoku-box { background-color: #ffaa0022; padding: 10px; border-radius: 10px; border-left: 5px solid #ffaa00; }
    .volume-box { background-color: #00ffaa22; padding: 10px; border-radius: 10px; border-left: 5px solid #00ffaa; }
    .pump-badge { background-color: #ff4444; color: white; padding: 5px 10px; border-radius: 20px; font-weight: bold; display: inline-block; }
    .trend-up { color: #00ff00; font-weight: bold; }
    .trend-down { color: #ff4444; font-weight: bold; }
    .trend-sideways { color: #ffaa00; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ======================== HEADER ========================
st.title("🔮 Crypto Scanner - Ultimate Edition (FINAL)")
st.markdown("""
<div class="medium-font">
<b>Fitur Super Lengkap:</b> Ichimoku + Whale Detection + Volume Analysis + Prediksi Trend<br>
📌 <i>Berdasarkan diskusi forum crypto: ambil yang baik, buang yang jelek</i>
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
def predict_trend_direction(df, has_volume_spike, spike_ratio):
    """
    Prediksi trend dengan mempertimbangkan volume spike
    Volume spike >2x = sinyal kuat untuk bullish
    """
    if df is None or len(df) < 50:
        return None
    
    try:
        current_price = df['close'].iloc[-1]
        last = df.iloc[-1]
        
        # ========== ICHIMOKU SIGNAL (Bobot 25%) ==========
        ichi_score = 0
        if last['close'] > last['senkou_a'] and last['close'] > last['senkou_b']:
            ichi_score += 2
        else:
            ichi_score -= 1
        
        if last['tenkan'] > last['kijun']:
            ichi_score += 1
        else:
            ichi_score -= 1
        
        if 'future_senkou_a' in df.columns and 'future_senkou_b' in df.columns:
            if df['future_senkou_a'].iloc[-1] > df['future_senkou_b'].iloc[-1]:
                ichi_score += 1
        
        # ========== TREND SIGNAL (Bobot 20%) ==========
        trend_score = 0
        if last['close'] > last['MA50']:
            trend_score += 1
        if last['close'] > last['MA200']:
            trend_score += 1
        if last['MACD'] > last['Signal']:
            trend_score += 1
        
        # ========== VOLUME SPIKE SIGNAL (Bobot 35%) - DIPERKUAT! ==========
        volume_score = 0
        volume_interpretation = ""
        
        if has_volume_spike:
            if spike_ratio >= 5:
                volume_score += 5  # SANGAT BESAR
                volume_interpretation = f"🔊 VOLUME SPIKE {spike_ratio}x (LUAR BIASA!) - Whale masuk besar-besaran! Potensi pump gede."
            elif spike_ratio >= 3:
                volume_score += 4
                volume_interpretation = f"🔥 VOLUME SPIKE {spike_ratio}x (SANGAT BESAR) - Ada whale masuk! Potensi pump besar."
            elif spike_ratio >= 2:
                volume_score += 3
                volume_interpretation = f"📊 Volume spike {spike_ratio}x - Whale mulai gerak."
            
            # Cek korelasi volume dengan harga
            last_vol_ratio = df['Volume_Ratio'].iloc[-1] if not pd.isna(df['Volume_Ratio'].iloc[-1]) else 1
            if last_vol_ratio > 1.5 and last['close'] > last['open']:
                volume_score += 1
                volume_interpretation += " Harga nutup hijau dengan volume tinggi = akumulasi bullish."
            elif last_vol_ratio > 1.5 and last['close'] < last['open']:
                volume_score -= 1
                volume_interpretation += " Harga nutup merah dengan volume tinggi = distribusi (hati-hati)."
        else:
            volume_interpretation = "Volume normal - Tidak ada whale terdeteksi."
        
        # ========== RSI SIGNAL (Bobot 10%) ==========
        rsi_score = 0
        if last['RSI'] < 30:
            rsi_score += 1
        elif last['RSI'] > 70:
            rsi_score -= 1
        elif 40 <= last['RSI'] <= 60:
            rsi_score += 0.5
        
        # ========== WHALE WICK SIGNAL (Bobot 10%) ==========
        wick_score = 0
        for i in range(1, 4):
            candle = df.iloc[-i]
            body = abs(candle['close'] - candle['open'])
            upper_wick = candle['high'] - max(candle['close'], candle['open'])
            lower_wick = min(candle['close'], candle['open']) - candle['low']
            candle_range = candle['high'] - candle['low']
            
            if candle_range > 0 and body > 0:
                if upper_wick > body * 2:
                    wick_score -= 1
                if lower_wick > body * 2:
                    wick_score += 1
        
        # ========== TOTAL SCORE ==========
        total_score = ichi_score + trend_score + volume_score + rsi_score + wick_score
        
        # Normalisasi ke persentase
        max_possible = 15
        confidence = min(95, max(30, (total_score + 8) / max_possible * 100))
        
        # Penentuan arah berdasarkan score (volume spike bisa override)
        if has_volume_spike and spike_ratio >= 3 and total_score >= -2:
            # Volume spike besar bisa override sinyal bearish
            final_direction = "BULLISH (Naik)"
            final_direction_icon = "📈"
        elif total_score >= 2:
            final_direction = "BULLISH (Naik)"
            final_direction_icon = "📈"
        elif total_score <= -2:
            final_direction = "BEARISH (Turun)"
            final_direction_icon = "📉"
        else:
            final_direction = "SIDEWAYS (Mendatar)"
            final_direction_icon = "➡️"
        
        # Estimasi target harga berdasarkan arah dan volume spike
        atr = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else current_price * 0.02
        
        if final_direction == "BULLISH (Naik)":
            if has_volume_spike and spike_ratio >= 5:
                multiplier = 3.0
                estimated_timeframe = "3-7 hari"
            elif has_volume_spike and spike_ratio >= 3:
                multiplier = 2.5
                estimated_timeframe = "3-7 hari"
            elif has_volume_spike:
                multiplier = 2.0
                estimated_timeframe = "1-2 minggu"
            else:
                multiplier = 1.2
                estimated_timeframe = "2-4 minggu"
            
            weighted_target = current_price + (atr * multiplier * 3)
            predicted_change = (weighted_target - current_price) / current_price * 100
            predicted_change = min(predicted_change, 70)
            
        elif final_direction == "BEARISH (Turun)":
            if wick_score <= -2:
                multiplier = 1.5
                estimated_timeframe = "3-7 hari"
            else:
                multiplier = 1.0
                estimated_timeframe = "1-3 minggu"
            
            weighted_target = current_price - (atr * multiplier * 2)
            predicted_change = (weighted_target - current_price) / current_price * 100
            predicted_change = max(predicted_change, -30)
            
        else:
            weighted_target = current_price
            predicted_change = 0
            estimated_timeframe = "Tidak jelas"
        
        return {
            'final_direction': final_direction,
            'final_direction_icon': final_direction_icon,
            'weighted_target': round(weighted_target, 8),
            'avg_confidence': round(confidence, 1),
            'estimated_timeframe': estimated_timeframe,
            'predicted_change_pct': round(predicted_change, 1),
            'volume_interpretation': volume_interpretation,
            'total_score': total_score,
            'ichi_score': ichi_score,
            'trend_score': trend_score,
            'volume_score': volume_score,
            'rsi_score': rsi_score,
            'wick_score': wick_score
        }
        
    except Exception as e:
        print(f"Error predict trend: {e}")
        return None

def get_ichimoku_summary(df):
    """Ringkasan Ichimoku yang informatif"""
    if df is None or len(df) < 52:
        return None
    
    try:
        last = df.iloc[-1]
        current_price = last['close']
        
        # Posisi terhadap Cloud
        senkou_a = last['senkou_a']
        senkou_b = last['senkou_b']
        
        if not pd.isna(senkou_a) and not pd.isna(senkou_b):
            cloud_top = max(senkou_a, senkou_b)
            cloud_bottom = min(senkou_a, senkou_b)
            cloud_thickness = (cloud_top - cloud_bottom) / cloud_bottom * 100 if cloud_bottom > 0 else 0
            
            if current_price > cloud_top:
                cloud_position = "DI ATAS Cloud"
                cloud_status = "✅ Bullish - Cloud jadi support"
            elif current_price < cloud_bottom:
                cloud_position = "DI BAWAH Cloud"
                cloud_status = "❌ Bearish - Cloud jadi resistance"
            else:
                cloud_position = "DI DALAM Cloud"
                cloud_status = "⚠️ Netral - Dalam konsolidasi"
            
            if cloud_thickness > 3:
                cloud_thickness_text = "TEBAL (Resistance kuat)" if current_price < cloud_bottom else "TEBAL (Support kuat)"
            elif cloud_thickness > 1:
                cloud_thickness_text = "SEDANG"
            else:
                cloud_thickness_text = "TIPIS (Mudah ditembus)"
        else:
            cloud_position = "Tidak tersedia"
            cloud_status = "Data tidak cukup"
            cloud_thickness = 0
            cloud_thickness_text = "N/A"
        
        # TK Cross
        tenkan = last['tenkan']
        kijun = last['kijun']
        if not pd.isna(tenkan) and not pd.isna(kijun):
            if tenkan > kijun:
                tk_status = "✅ Golden Cross (Tenkan > Kijun) - Bullish"
            else:
                tk_status = "⚠️ Death Cross (Tenkan < Kijun) - Bearish"
        else:
            tk_status = "Tidak tersedia"
        
        # Chikou Span
        chikou = last['chikou'] if 'chikou' in df.columns else None
        price_26_ago = df['close'].iloc[-26] if len(df) >= 26 else None
        if chikou is not None and price_26_ago is not None and not pd.isna(chikou):
            if chikou > price_26_ago:
                chikou_status = "✅ Chikou di atas harga (konfirmasi bullish)"
            else:
                chikou_status = "❌ Chikou di bawah harga (konfirmasi bearish)"
        else:
            chikou_status = "Tidak tersedia"
        
        # Future Cloud
        future_a = df['future_senkou_a'].iloc[-1] if 'future_senkou_a' in df.columns else None
        future_b = df['future_senkou_b'].iloc[-1] if 'future_senkou_b' in df.columns else None
        if future_a is not None and future_b is not None and not pd.isna(future_a) and not pd.isna(future_b):
            if future_a > future_b:
                future_status = "🔮 Future Cloud: BULLISH (Hijau)"
            else:
                future_status = "🔮 Future Cloud: BEARISH (Merah)"
        else:
            future_status = "Tidak tersedia"
        
        return {
            'cloud_position': cloud_position,
            'cloud_status': cloud_status,
            'cloud_thickness': round(cloud_thickness, 1),
            'cloud_thickness_text': cloud_thickness_text,
            'tk_status': tk_status,
            'chikou_status': chikou_status,
            'future_status': future_status
        }
        
    except Exception as e:
        print(f"Error get ichimoku summary: {e}")
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
                    breakout_prediction = "✅ Harga DI ATAS garis trend naik -> support dinamis, potensi lanjut naik"
                else:
                    breakout_prediction = "⚠️ Harga MENDEXATI garis trend naik -> potensi rebound dalam 2-5 hari"
        
        if trendline_down and trendline_down['direction'] == 'down':
            distance_to_resistance = abs(trendline_down['current_value'] - current_price) / current_price * 100
            if distance_to_resistance < 2:
                if current_price > trendline_down['current_value']:
                    breakout_prediction = "🚀 Harga BREAKOUT garis trend turun -> bullish signal!"
                else:
                    breakout_prediction = "⚠️ Harga di BAWAH garis trend turun -> resistance dinamis, butuh volume besar"
        
        return {
            'trendline_up': trendline_up,
            'trendline_down': trendline_down,
            'breakout_prediction': breakout_prediction
        }
        
    except Exception as e:
        print(f"Error detect trendline: {e}")
        return None

def predict_breakout_time(df, current_price, nearest_resistance, nearest_support, trend_prediction=None):
    if df is None or len(df) < 30:
        return None
    
    try:
        distance_to_resistance = (nearest_resistance - current_price) / current_price * 100 if nearest_resistance > current_price else 0
        distance_to_support = (current_price - nearest_support) / current_price * 100 if current_price > nearest_support else 0
        
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else current_price * 0.02
        
        if distance_to_resistance > 0 and atr > 0:
            days_to_resistance = max(1, int(distance_to_resistance / (atr / current_price * 100)))
        else:
            days_to_resistance = 999
        
        vol_trend = df['Volume_Ratio'].tail(5).mean()
        
        # CEK KONSISTENSI DENGAN PREDIKSI TREND
        is_trend_bullish = trend_prediction and trend_prediction['final_direction'] == "BULLISH (Naik)"
        is_trend_bearish = trend_prediction and trend_prediction['final_direction'] == "BEARISH (Turun)"
        
        if is_trend_bearish:
            breakout_soon = False
            estimated_days = days_to_resistance if days_to_resistance < 999 else 7
            breakout_type = "NONE (Bearish Trend)"
            breakout_price = None
            advice = "⏸️ HOLD DULU - Tunggu harga turun ke support yang lebih dalam"
        elif is_trend_bullish:
            if vol_trend > 1.5 and distance_to_resistance < 5:
                breakout_soon = True
                estimated_days = min(3, days_to_resistance)
                breakout_type = "UP (Resistance)"
                breakout_price = nearest_resistance
                advice = f"⚡ Potensi breakout dalam {estimated_days} hari ke {breakout_price:.6f}"
            elif distance_to_resistance < 3:
                breakout_soon = True
                estimated_days = days_to_resistance
                breakout_type = "UP (Resistance)"
                breakout_price = nearest_resistance
                advice = f"⚠️ Harga mendekati resistance {breakout_price:.6f}, siap-siap breakout"
            else:
                breakout_soon = False
                estimated_days = days_to_resistance if days_to_resistance < 999 else 7
                breakout_type = "Unknown"
                breakout_price = None
                advice = f"Jarak ke resistance {distance_to_resistance:.1f}%, perlu volume untuk breakout"
        else:
            # Sideways
            if vol_trend > 1.5 and distance_to_resistance < 5:
                breakout_soon = True
                estimated_days = min(3, days_to_resistance)
                breakout_type = "UP (Resistance)"
                breakout_price = nearest_resistance
                advice = f"⚡ Potensi breakout dalam {estimated_days} hari ke {breakout_price:.6f}"
            elif distance_to_support < 3:
                breakout_soon = True
                estimated_days = 1
                breakout_type = "DOWN (Support)"
                breakout_price = nearest_support
                advice = f"⚠️ Harga mendekati support {breakout_price:.6f}, potensi breakdown"
            else:
                breakout_soon = False
                estimated_days = days_to_resistance if days_to_resistance < 999 else 7
                breakout_type = "Unknown"
                breakout_price = None
                advice = "Belum ada sinyal breakout yang jelas"
        
        return {
            'breakout_soon': breakout_soon,
            'estimated_days': estimated_days,
            'breakout_type': breakout_type,
            'breakout_price': breakout_price,
            'distance_to_resistance_pct': round(distance_to_resistance, 2),
            'distance_to_support_pct': round(distance_to_support, 2),
            'volume_trend': round(vol_trend, 1),
            'advice': advice
        }
        
    except Exception as e:
        print(f"Error predict breakout: {e}")
        return None

def predict_candlestick_pattern(df, days_ahead=5):
    if df is None or len(df) < 30:
        return None
    
    try:
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        volume_ratio = df['Volume_Ratio'].iloc[-1] if 'Volume_Ratio' in df.columns else 1
        last = df.iloc[-1]
        
        predictions = []
        
        # Deteksi pola terakhir
        body = abs(last['close'] - last['open'])
        upper_wick = last['high'] - max(last['close'], last['open'])
        lower_wick = min(last['close'], last['open']) - last['low']
        candle_range = last['high'] - last['low']
        
        if candle_range > 0:
            wick_ratio = (upper_wick + lower_wick) / candle_range
            
            if wick_ratio > 0.6 and body / candle_range < 0.2:
                predictions.append("📊 Candle terakhir: DOJI - Indecision, harga bisa balik arah")
            elif upper_wick > body * 2:
                predictions.append("📊 Candle terakhir: SHOOTING STAR - Potensi reversal bearish")
            elif lower_wick > body * 2:
                predictions.append("📊 Candle terakhir: HAMMER - Potensi reversal bullish")
        
        # Prediksi berdasarkan RSI
        if rsi < 30:
            predictions.append(f"Hari 1-2: Potensi reversal BULLISH karena RSI oversold {rsi:.0f}")
        elif rsi > 70:
            predictions.append(f"Hari 1-2: Potensi reversal BEARISH karena RSI overbought {rsi:.0f}")
        
        # Prediksi berdasarkan volume
        if volume_ratio > 2:
            predictions.append(f"Hari 1: Potensi VOLUME SPIKE + pergerakan besar")
        
        # Prediksi berdasarkan squeeze
        bb_width = df['BB_Width'].iloc[-1] if 'BB_Width' in df.columns else 0.1
        if bb_width < 0.05:
            predictions.append(f"Hari 2-3: Potensi VOLATILITY EXPANSION (squeeze {bb_width*100:.1f}%)")
        
        # Prediksi arah umum
        macd_hist = df['MACD_Hist'].iloc[-3:].values if 'MACD_Hist' in df.columns else [0,0,0]
        if len(macd_hist) >= 3:
            if macd_hist[-1] > macd_hist[-2] > macd_hist[-3]:
                predictions.append("Hari 3-5: Cenderung GREEN CANDLE (momentum bullish)")
            elif macd_hist[-1] < macd_hist[-2] < macd_hist[-3]:
                predictions.append("Hari 3-5: Cenderung RED CANDLE (momentum bearish)")
        
        if not predictions:
            predictions.append("Belum ada sinyal pola candlestick yang jelas")
        
        return predictions
        
    except Exception as e:
        print(f"Error predict candlestick: {e}")
        return ["Tidak bisa memprediksi pola candlestick"]

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
            return True, "bullish", f"🔥 OBV naik {obv_slope:.0f} tapi harga turun {abs(price_slope):.0f} -> WHALE BELI DIAM-DIAM! (Akumulasi)"
        elif obv_slope < 0 and price_slope > 0:
            return True, "bearish", f"⚠️ OBV turun {abs(obv_slope):.0f} tapi harga naik {price_slope:.0f} -> WHALE JUAL DIAM-DIAM! (Distribusi)"
        return False, "neutral", "Tidak ada divergensi OBV"
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
            if upper_wick > body * 2:
                wick_to_price_ratio = upper_wick / current_price * 100
                manipulations.append(f"Candle {i}: 🔴 Wick ATAS ({upper_wick/body:.1f}x body) - whale jual, potensi turun {wick_to_price_ratio:.1f}%")
                score -= 1
            if lower_wick > body * 2:
                wick_to_price_ratio = lower_wick / current_price * 100
                manipulations.append(f"Candle {i}: 🟢 Wick BAWAH ({lower_wick/body:.1f}x body) - whale beli, potensi naik {wick_to_price_ratio:.1f}%")
                score += 1
    
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
                return True, round(concentration * 100, 1), f"🎯 Volume terkonsentrasi {concentration*100:.1f}% di range {max_bin} - whale akumulasi di level itu!"
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
            large_tx.append(f"🔊 Candle {i}: Volume {candle_vol/avg_vol:.1f}x normal - transaksi besar!")
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
            fake_signals.append(f"⚠️ Breakout volume RENDAH ({vol_ratio:.1f}x) -> FAKEOUT! Harga bisa balik turun")
            fake_score += 2
    
    upper_wick = last['high'] - max(last['close'], last['open'])
    body = abs(last['close'] - last['open'])
    if body > 0 and upper_wick > body * 2 and current_price > nearest_resistance:
        fake_signals.append(f"⚠️ Wick panjang {upper_wick/body:.1f}x body di resistance -> whale jual, FAKEOUT!")
        fake_score += 2
    
    if prev['high'] > nearest_resistance and prev['close'] < nearest_resistance:
        fake_signals.append("⚠️ Candle sebelumnya tembus tapi nutup di bawah -> FALSE BREAKOUT!")
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
            return True, f"✅ Akumulasi: sideways {price_range:.1%}, volume turun {avg_vol_ratio:.2f}x, OBV naik {df['OBV_change_pct'].iloc[-1]:.1f}% - Whale sedang kumpulin coin!"
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

def get_entry_stop_loss(df, current_price, nearest_support, trend_direction, has_volume_spike, spike_ratio):
    if df is None or len(df) < 20:
        return current_price * 0.95, current_price * 0.90
    
    atr = df['ATR'].iloc[-1] if not pd.isna(df['ATR'].iloc[-1]) else current_price * 0.02
    
    if trend_direction == "BULLISH (Naik)":
        # Entry di support atau di atas support
        if has_volume_spike and spike_ratio >= 3:
            # Volume spike besar -> entry lebih agresif
            conservative_entry = current_price * 0.98
        else:
            conservative_entry = nearest_support if nearest_support and nearest_support < current_price else current_price * 0.97
        
        recent_lows = df['low'].tail(20)
        swing_low = recent_lows.min()
        stop_loss = swing_low * 0.98 if swing_low < conservative_entry else conservative_entry * 0.95
        
    elif trend_direction == "BEARISH (Turun)":
        conservative_entry = current_price * 0.92
        stop_loss = conservative_entry * 0.95
    else:
        # Sideways
        conservative_entry = nearest_support if nearest_support and nearest_support < current_price else current_price * 0.96
        stop_loss = conservative_entry * 0.95
    
    if conservative_entry - stop_loss > conservative_entry * 0.15:
        stop_loss = conservative_entry * 0.85
    
    return round(conservative_entry, 8), round(stop_loss, 8)

def calculate_targets_by_trend(entry_price, atr, trend_direction, has_volume_spike, spike_ratio):
    """
    TARGET PROFIT SESUAI ARAH TREND dengan mempertimbangkan volume spike
    """
    if trend_direction == "BULLISH (Naik)":
        # Volume spike memperkuat target
        if has_volume_spike and spike_ratio >= 5:
            multiplier = 4.0
            target_type = "Take Profit (Jual di harga tinggi) - PUMP BESAR!"
        elif has_volume_spike and spike_ratio >= 3:
            multiplier = 3.5
            target_type = "Take Profit (Jual di harga tinggi) - POTENSI PUMP"
        elif has_volume_spike:
            multiplier = 2.5
            target_type = "Take Profit (Jual di harga tinggi)"
        else:
            multiplier = 2.0
            target_type = "Take Profit (Jual di harga tinggi) - Konservatif"
        
        tp1 = entry_price + (atr * multiplier)
        tp2 = entry_price + (atr * multiplier * 1.8)
        tp3 = entry_price + (atr * multiplier * 3.0)
        
        tp1_pct = ((tp1 - entry_price) / entry_price) * 100
        tp2_pct = ((tp2 - entry_price) / entry_price) * 100
        tp3_pct = ((tp3 - entry_price) / entry_price) * 100
        
    elif trend_direction == "BEARISH (Turun)":
        # Target untuk buy zone (beli di harga lebih rendah)
        multiplier = 1.5
        target_type = "Buy Zone (Beli di harga lebih rendah - DCA)"
        
        tp1 = entry_price - (atr * multiplier)
        tp2 = entry_price - (atr * multiplier * 2)
        tp3 = entry_price - (atr * multiplier * 3.5)
        
        # Batasi maksimum -30%
        tp1 = max(tp1, entry_price * 0.85)
        tp2 = max(tp2, entry_price * 0.75)
        tp3 = max(tp3, entry_price * 0.65)
        
        tp1_pct = ((tp1 - entry_price) / entry_price) * 100
        tp2_pct = ((tp2 - entry_price) / entry_price) * 100
        tp3_pct = ((tp3 - entry_price) / entry_price) * 100
        
    else:
        target_type = "Target Konservatif"
        tp1 = entry_price + (atr * 1.5)
        tp2 = entry_price + (atr * 2.5)
        tp3 = entry_price + (atr * 4)
        tp1_pct = ((tp1 - entry_price) / entry_price) * 100
        tp2_pct = ((tp2 - entry_price) / entry_price) * 100
        tp3_pct = ((tp3 - entry_price) / entry_price) * 100
    
    return round(tp1, 8), round(tp2, 8), round(tp3, 8), round(tp1_pct, 1), round(tp2_pct, 1), round(tp3_pct, 1), target_type

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
                return True, "⚠️ Head and Shoulders (bearish reversal) - Harga berpotensi turun!"
        
        return False, ""
    except:
        return False, ""

# ======================== FUNGSI UTAMA ANALISIS ========================
def analyze_coin_spot(symbol, exchange_name):
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
        
        # PREDIKSI TREND (dengan volume spike)
        trend_prediction = predict_trend_direction(daily, has_volume_spike, spike_ratio)
        
        if trend_prediction:
            final_trend = trend_prediction['final_direction']
        else:
            final_trend = "SIDEWAYS (Mendatar)"
        
        # Entry & Stop Loss (dengan mempertimbangkan volume spike)
        conservative_entry, stop_loss = get_entry_stop_loss(daily, current_price, nearest_support, final_trend, has_volume_spike, spike_ratio)
        atr_value = last['ATR'] if not pd.isna(last['ATR']) else current_price * 0.02
        
        # TARGET PROFIT SESUAI TREND
        tp1, tp2, tp3, tp1_pct, tp2_pct, tp3_pct, target_type = calculate_targets_by_trend(
            conservative_entry, atr_value, final_trend, has_volume_spike, spike_ratio
        )
        
        # Risk/Reward (hanya untuk bullish)
        if final_trend == "BULLISH (Naik)":
            risk = conservative_entry - stop_loss
            reward = tp1 - conservative_entry
            rr_ratio = reward / risk if risk > 0 else 0
        else:
            rr_ratio = 0
        
        # WHALE DETECTION
        has_divergence, div_type, div_reason = detect_obv_divergence(daily)
        wick_signals, wick_score = detect_wick_manipulation(daily)
        has_vol_concentration, vol_conc_pct, vol_profile_reason = detect_volume_profile_whale(daily)
        large_tx, tx_count = detect_large_transactions(daily)
        is_fakeout, fake_score, fake_signals = detect_fakeout(daily, current_price, nearest_resistance)
        
        # Ichimoku summary
        ichimoku_summary = get_ichimoku_summary(daily)
        
        # Trendline & Breakout
        trendline_analysis = detect_trendline(daily, lookback=50)
        breakout_prediction = predict_breakout_time(daily, current_price, nearest_resistance, nearest_support, trend_prediction)
        candlestick_prediction = predict_candlestick_pattern(daily, days_ahead=5)
        
        # Whale confidence
        whale_confidence = 0
        whale_signals = []
        
        if has_divergence and div_type == "bullish":
            whale_confidence += 3
            whale_signals.append(f"🐋 {div_reason}")
        if wick_score > 0:
            whale_confidence += wick_score
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
            for fs in fake_signals[:1]:
                whale_signals.append(f"⚠️ {fs}")
        
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
            reasons.append("✅ Harga di atas MA200 (bullish jangka panjang)")
        elif last['close'] > last['MA50']:
            score += 1
            reasons.append("📈 Harga di atas MA50 (trend naik)")
        else:
            reasons.append("⚠️ Harga di bawah MA50 (trend bearish)")
        
        above_cloud = last['close'] > last['senkou_a'] and last['close'] > last['senkou_b']
        if above_cloud:
            score += 2
            reasons.append("✅ Harga di atas Cloud Ichimoku (bullish)")
        else:
            reasons.append("⚠️ Harga di bawah Cloud Ichimoku (bearish)")
        
        tk_cross = last['tenkan'] > last['kijun']
        if tk_cross:
            score += 1
            reasons.append("✅ TK Cross (golden cross) - momentum positif")
        else:
            reasons.append("⚠️ TK Cross (death cross) - momentum negatif")
        
        if 40 <= last['RSI'] <= 60:
            score += 1
            reasons.append(f"📊 RSI {last['RSI']:.1f} (netral)")
        elif last['RSI'] < 30:
            reasons.append(f"💀 RSI {last['RSI']:.1f} (oversold) - potensi reversal")
        elif last['RSI'] > 70:
            reasons.append(f"⚠️ RSI {last['RSI']:.1f} (overbought) - hati-hati koreksi")
        else:
            score += 0.5
            reasons.append(f"📊 RSI {last['RSI']:.1f}")
        
        if last['MACD'] > last['Signal']:
            score += 1
            reasons.append("✅ MACD bullish (momentum positif)")
        else:
            reasons.append("📉 MACD bearish (momentum negatif)")
        
        if has_volume_spike:
            score += 3  # Bobot lebih besar!
            reasons.append(f"🔊 Volume spike {spike_ratio}x! - Ada whale masuk besar!")
        elif last['Volume_Ratio'] < 0.7:
            score += 0.5
            reasons.append(f"🔇 Volume rendah {last['Volume_Ratio']:.2f}x (potensi akumulasi)")
        
        if is_squeeze:
            score += 1.5
            reasons.append(f"🔥 Volatility Squeeze {squeeze_pct}%! - Harga siap meledak!")
        
        if has_breakout:
            score += 1.5
            reasons.append(f"🚀 Breakout resistance di {breakout_level:.8f}!")
        
        if detect_double_bottom(daily):
            score += 1
            reasons.append("📉 Pola Double Bottom (reversal bullish)")
        if detect_bullish_flag(daily):
            score += 1
            reasons.append("🚩 Pola Bullish Flag (lanjutan uptrend)")
        if detect_cup_handle(daily):
            score += 1
            reasons.append("🏆 Pola Cup and Handle (bullish klasik)")
        
        hs_detected, hs_reason = detect_head_shoulders(daily)
        if hs_detected:
            score -= 1
            reasons.append(f"⚠️ {hs_reason}")
        
        if daily['OBV_trend'].iloc[-1]:
            score += 1
            reasons.append(f"🐋 OBV naik {daily['OBV_change_pct'].iloc[-1]:.1f}% - whale akumulasi!")
        
        # STATUS (prioritaskan volume spike)
        if has_volume_spike and spike_ratio >= 3 and final_trend != "BEARISH (Turun)":
            status = "🚀 POTENSI PUMP - WHALE MASUK!"
            confidence = "Tinggi"
            action = "BELI (Agresif)"
        elif final_trend == "BEARISH (Turun)":
            status = "⚠️ HOLD / WAIT - TREND BEARISH"
            confidence = "Rendah"
            action = "HOLD"
        elif score >= 8:
            status = "🚀 STRONG BUY - SIAP PUMP!"
            confidence = "Tinggi"
            action = "BELI"
        elif score >= 6:
            status = "📈 BUY - POTENSI PUMP"
            confidence = "Sedang"
            action = "BELI"
        elif score >= 4:
            status = "⏳ ACCUMULATION - PANTAU"
            confidence = "Rendah"
            action = "PANTAU"
        else:
            status = "⏸️ HOLD / WAIT"
            confidence = "Sangat Rendah"
            action = "HOLD"
        
        # Potensi pump
        if trend_prediction:
            pump_potential = f"{abs(trend_prediction['predicted_change_pct'])}% dalam {trend_prediction['estimated_timeframe']}"
        else:
            pump_potential = "Tidak terprediksi"
        
        # ENTRY TIMING
        entry_timing_status = "⏳ PANTAU"
        entry_timing_reason = ""
        
        if final_trend == "BEARISH (Turun)":
            entry_timing_status = "⏸️ HOLD DULU"
            entry_timing_reason = f"Prediksi BEARISH ({trend_prediction['avg_confidence'] if trend_prediction else 50:.0f}% confidence). Hindari beli, tunggu harga turun ke support yang lebih dalam."
        elif final_trend == "BULLISH (Naik)":
            if breakout_prediction and breakout_prediction['breakout_soon']:
                entry_timing_status = "⚡ AGGRESSIVE ENTRY"
                entry_timing_reason = breakout_prediction['advice']
            elif conservative_entry < current_price:
                entry_timing_status = "⬇️ TUNGGU PULLBACK"
                entry_timing_reason = f"Pasang limit order di {conservative_entry:.6f}"
            elif conservative_entry > current_price:
                entry_timing_status = "✅ BELI SEKARANG"
                entry_timing_reason = f"Harga {current_price:.6f} sudah di bawah entry dalam trend bullish"
            else:
                entry_timing_status = "⏳ PASANG LIMIT"
                entry_timing_reason = f"Pasang limit order di {conservative_entry:.6f}"
        else:
            # Sideways
            if has_volume_spike and spike_ratio >= 3:
                entry_timing_status = "⚡ AGGRESSIVE ENTRY"
                entry_timing_reason = f"Volume spike {spike_ratio}x! Potensi breakout besar, bisa entry sekarang dengan stop loss ketat."
            elif breakout_prediction and breakout_prediction['breakout_soon']:
                entry_timing_status = "⚡ AGGRESSIVE ENTRY"
                entry_timing_reason = breakout_prediction['advice']
            elif conservative_entry < current_price:
                entry_timing_status = "⬇️ TUNGGU PULLBACK"
                entry_timing_reason = f"Pasang limit order di {conservative_entry:.6f}"
            elif conservative_entry > current_price:
                entry_timing_status = "✅ BELI SEKARANG"
                entry_timing_reason = f"Harga {current_price:.6f} sudah di bawah entry"
            else:
                entry_timing_status = "⏳ PASANG LIMIT"
                entry_timing_reason = f"Pasang limit order di {conservative_entry:.6f}"
        
        beginner_summary = f"""
        **Gampangnya:** 
        - 📊 Skor {score:.1f}/10 → {status}
        - 🔮 Prediksi trend: {final_trend} ({trend_prediction['avg_confidence'] if trend_prediction else 0}% confidence)
        - ⏰ Timing: {entry_timing_status} - {entry_timing_reason}
        - 💰 Entry: {conservative_entry:.6f}
        - 🛑 Stop loss: {stop_loss:.6f}
        """
        
        if final_trend == "BULLISH (Naik)":
            beginner_summary += f"\n- 🎯 {target_type}: {tp1:.6f} (+{tp1_pct:.1f}%) → {tp2:.6f} (+{tp2_pct:.1f}%) → {tp3:.6f} (+{tp3_pct:.1f}%)"
        elif final_trend == "BEARISH (Turun)":
            beginner_summary += f"\n- 🎯 {target_type}: {tp1:.6f} ({tp1_pct:.1f}%) → {tp2:.6f} ({tp2_pct:.1f}%) → {tp3:.6f} ({tp3_pct:.1f}%)"
            beginner_summary += f"\n- 💡 Catatan: Target adalah level entry yang lebih rendah (DCA), BUKAN take profit!"
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 8),
            'entry_price': conservative_entry,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'tp1_pct': tp1_pct,
            'tp2_pct': tp2_pct,
            'tp3_pct': tp3_pct,
            'target_type': target_type,
            'rr_ratio': round(rr_ratio, 2) if final_trend == "BULLISH (Naik)" else 0,
            'score': round(score, 1),
            'status': status,
            'action': action,
            'confidence': confidence,
            'pump_potential': pump_potential,
            'is_accum': is_accum,
            'has_volume_spike': has_volume_spike,
            'spike_ratio': spike_ratio,
            'is_squeeze': is_squeeze,
            'squeeze_pct': squeeze_pct,
            'has_breakout': has_breakout,
            'breakout_level': breakout_level,
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
            'ichimoku_summary': ichimoku_summary,
            'trendline_analysis': trendline_analysis,
            'breakout_prediction': breakout_prediction,
            'candlestick_prediction': candlestick_prediction,
            'entry_timing_status': entry_timing_status,
            'entry_timing_reason': entry_timing_reason,
            'is_fakeout': is_fakeout,
            'has_divergence': has_divergence,
            'final_trend': final_trend
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
        
        # Deteksi volume spike cepat
        max_vol = daily['Volume_Ratio'].tail(5).max()
        if max_vol > 2:
            score += 2
        
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
                        subplot_titles=(f"{symbol} - Harga + Ichimoku + Garis Trend", "Volume", "RSI"))
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Harga'
    ), row=1, col=1)
    
    # Ichimoku Cloud
    if 'senkou_a' in df.columns and 'senkou_b' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_a'], 
                                name='Senkou A', line=dict(color='green', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['senkou_b'], 
                                name='Senkou B', line=dict(color='red', width=1, dash='dash')), row=1, col=1)
        
        # Fill cloud
        fig.add_trace(go.Scatter(
            x=pd.concat([df['timestamp'], df['timestamp'][::-1]]),
            y=pd.concat([df['senkou_a'], df['senkou_b'][::-1]]),
            fill='toself', fillcolor='rgba(0,100,0,0.2)', 
            line=dict(color='rgba(0,0,0,0)'), name='Cloud'
        ), row=1, col=1)
    
    # Tenkan & Kijun
    if 'tenkan' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['tenkan'], 
                                name='Tenkan (9)', line=dict(color='blue', width=1)), row=1, col=1)
    if 'kijun' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['kijun'], 
                                name='Kijun (26)', line=dict(color='orange', width=1)), row=1, col=1)
    
    # Moving Averages
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA50'], 
                            name='MA50', line=dict(color='purple', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['MA200'], 
                            name='MA200', line=dict(color='brown', width=1.5)), row=1, col=1)
    
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
    
    # Volume dengan warna
    colors = []
    for i, row in df.iterrows():
        if row['close'] > row['open']:
            colors.append('green')
        else:
            colors.append('red')
    
    # Highlight volume spike
    vol_ma = df['Volume_MA20'].fillna(df['volume'].mean())
    spike_indicator = df['volume'] > vol_ma * 2
    
    bar_colors = ['orange' if spike_indicator.iloc[i] else colors[i] for i in range(len(df))]
    
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color=bar_colors), row=2, col=1)
    
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
    st.header("📖 Fitur Super Lengkap")
    st.markdown("""
    **🔮 Prediksi Trend** - 5 komponen (Ichimoku, Trend, Volume, RSI, Whale)
    **☁️ Ichimoku Cloud** - Posisi Cloud, TK Cross, Chikou, Future Cloud
    **🐋 Whale Detection** - OBV Divergence, Volume Profile, Wick Manipulation
    **📊 Volume Analysis** - Deteksi volume spike dan akumulasi
    **📐 Garis Trend** - Support/resistance dinamis
    **⚡ Prediksi Breakout** - Kapan dan di harga berapa
    **🕯️ Pola Candlestick** - Prediksi 5 hari ke depan
    
    **📌 Target Profit:**
    - BULLISH → Target Take Profit (harga naik) - lebih besar jika volume spike
    - BEARISH → Target Buy Zone (harga turun untuk entry)
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
        if row['score'] >= 8:
            bg = "#00ff0015"
            badge = "🚀 SIAP PUMP"
        elif row['score'] >= 6:
            bg = "#44ff0015"
            badge = "📈 POTENSI PUMP"
        elif row['score'] >= 4:
            bg = "#ffaa0015"
            badge = "⏳ AKUMULASI"
        else:
            bg = "#ff444415"
            badge = "⚠️ HOLD"
        
        st.markdown(f"""
        <div style="background-color: {bg}; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
            <table style="width: 100%;">
                <tr><td style="width: 25%;"><b>{row['symbol']}</b></td>
                    <td style="width: 15%;">💰 ${row['price']:.6f}</td>
                    <td style="width: 15%;">⭐ Skor: {row['score']}</td>
                    <td style="width: 25%;"><span class="pump-badge">{badge}</span></td>
                    <td style="width: 20%;">{'🐋 OBV Naik' if row['obv_trend'] else ''}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("🔍 Analisis Detail")
    selected_coin = st.selectbox("Pilih coin", df['symbol'].tolist())
    
    if selected_coin:
        with st.spinner(f"Menganalisis {selected_coin}..."):
            result = analyze_coin_spot(selected_coin, exchange_name)
            
            if result:
                detail = result
                
                # Status box
                if detail['final_trend'] == "BEARISH (Turun)":
                    box_class = "avoid-box"
                elif detail['score'] >= 8:
                    box_class = "buy-box"
                elif detail['score'] >= 6:
                    box_class = "buy-box"
                elif detail['score'] >= 4:
                    box_class = "wait-box"
                else:
                    box_class = "avoid-box"
                
                st.markdown(f"""
                <div class="{box_class}">
                    <h3>{detail['status']}</h3>
                    <p><b>Skor:</b> {detail['score']}/10 | <b>Konfidensi:</b> {detail['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # VOLUME SPIKE BOX
                if detail['has_volume_spike']:
                    st.markdown(f"""
                    <div class="volume-box">
                        <h4>🔊 VOLUME SPIKE {detail['spike_ratio']}x!</h4>
                        <p>{detail['trend_prediction']['volume_interpretation'] if detail['trend_prediction'] else 'Ada lonjakan volume signifikan!'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ICHIMOKU BOX
                if detail['ichimoku_summary']:
                    ichi = detail['ichimoku_summary']
                    st.markdown(f"""
                    <div class="ichimoku-box">
                        <h4>☁️ Ichimoku Cloud Analysis</h4>
                        <table style="width: 100%;">
                            <tr><td style="width: 40%;"><b>Posisi Harga:</b></td>
                            <td>{ichi['cloud_position']} - {ichi['cloud_status']}</td>
                            </tr>
                            <tr><td><b>Ketebalan Cloud:</b></td>
                            <td>{ichi['cloud_thickness_text']} ({ichi['cloud_thickness']}%)</td>
                            </tr>
                            <tr><td><b>TK Cross:</b></td>
                            <td>{ichi['tk_status']}</td>
                            </tr>
                            <tr><td><b>Chikou Span:</b></td>
                            <td>{ichi['chikou_status']}</td>
                            </tr>
                            <tr><td><b>Future Cloud:</b></td>
                            <td>{ichi['future_status']}</td>
                            </tr>
                        </table>
                    </div>
                    """, unsafe_allow_html=True)
                
                # PREDIKSI TREND BOX
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
                    
                    # Detail score components
                    with st.expander("📊 Detail Prediksi (5 Komponen)"):
                        st.write(f"- Ichimoku Score: {tp['ichi_score']}")
                        st.write(f"- Trend Score: {tp['trend_score']}")
                        st.write(f"- Volume Score: {tp['volume_score']}")
                        st.write(f"- RSI Score: {tp['rsi_score']}")
                        st.write(f"- Whale Wick Score: {tp['wick_score']}")
                        st.write(f"- **Total Score: {tp['total_score']}**")
                
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
                        st.write(f"- Jarak ke resistance: {bp['distance_to_resistance_pct']}%")
                        st.write(f"- Jarak ke support: {bp['distance_to_support_pct']}%")
                        st.write(f"- Volume trend: {bp['volume_trend']}x normal")
                        if bp['breakout_soon']:
                            st.warning(f"🚨 {bp['advice']}")
                        else:
                            st.info(f"ℹ️ {bp['advice']}")
                
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
                    for ws in detail['whale_signals'][:4]:
                        st.write(f"- {ws}")
                
                # ENTRY TIMING
                if detail['entry_timing_status'] == "✅ BELI SEKARANG":
                    st.success(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
                elif detail['entry_timing_status'] == "⬇️ TUNGGU PULLBACK":
                    st.warning(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
                elif detail['entry_timing_status'] == "⚡ AGGRESSIVE ENTRY":
                    st.info(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
                elif detail['entry_timing_status'] == "⏸️ HOLD DULU":
                    st.error(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
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
                    if detail['final_trend'] == "BULLISH (Naik)":
                        st.metric("📊 Risk/Reward", f"1:{detail['rr_ratio']:.1f}")
                    else:
                        st.metric("📊 Risk/Reward", "N/A (Bearish)")
                
                # TARGET PROFIT
                st.write(f"**🎯 {detail['target_type']}:**")
                col1, col2, col3 = st.columns(3)
                
                if detail['final_trend'] == "BULLISH (Naik)":
                    with col1:
                        st.info(f"Target 1: ${detail['tp1']:.6f} (+{detail['tp1_pct']:.1f}%)")
                    with col2:
                        st.info(f"Target 2: ${detail['tp2']:.6f} (+{detail['tp2_pct']:.1f}%)")
                    with col3:
                        st.success(f"Target 3: ${detail['tp3']:.6f} (+{detail['tp3_pct']:.1f}%)")
                elif detail['final_trend'] == "BEARISH (Turun)":
                    with col1:
                        st.warning(f"Zone 1: ${detail['tp1']:.6f} ({detail['tp1_pct']:.1f}%)")
                    with col2:
                        st.warning(f"Zone 2: ${detail['tp2']:.6f} ({detail['tp2_pct']:.1f}%)")
                    with col3:
                        st.warning(f"Zone 3: ${detail['tp3']:.6f} ({detail['tp3_pct']:.1f}%)")
                    st.caption("💡 Ini adalah level BELI (DCA), BUKAN take profit. Harga diprediksi turun dulu.")
                else:
                    with col1:
                        st.info(f"Target 1: ${detail['tp1']:.6f} ({detail['tp1_pct']:+.1f}%)")
                    with col2:
                        st.info(f"Target 2: ${detail['tp2']:.6f} ({detail['tp2_pct']:+.1f}%)")
                    with col3:
                        st.info(f"Target 3: ${detail['tp3']:.6f} ({detail['tp3_pct']:+.1f}%)")
                
                st.info(detail['beginner_summary'])
                st.write(f"**📊 Support:** ${detail['nearest_support']:.6f} | **Resistance:** ${detail['nearest_resistance']:.6f}")
                
                # Signal badges
                st.write("**📡 Signal Terdeteksi:**")
                sig_cols = st.columns(7)
                with sig_cols[0]:
                    if detail['whale_confidence'] >= 3:
                        st.success("🐋 Whale")
                    else:
                        st.info("⏳ No Whale")
                with sig_cols[1]:
                    if detail['has_volume_spike']:
                        st.warning(f"🔊 Volume {detail['spike_ratio']}x")
                    else:
                        st.info("📊 Volume Normal")
                with sig_cols[2]:
                    if detail['is_squeeze']:
                        st.error(f"🔥 Squeeze {detail['squeeze_pct']}%")
                    else:
                        st.info("📈 No Squeeze")
                with sig_cols[3]:
                    if detail['has_breakout']:
                        st.success("🚀 Breakout")
                    else:
                        st.info("🔒 No Breakout")
                with sig_cols[4]:
                    if detail['tk_cross']:
                        st.success("📈 TK Cross")
                    else:
                        st.info("⚠️ No TK Cross")
                with sig_cols[5]:
                    if detail['has_divergence']:
                        st.success("🐋 Divergence")
                    else:
                        st.info("📊 No Divergence")
                with sig_cols[6]:
                    if detail['is_accum']:
                        st.success("📦 Akumulasi")
                    else:
                        st.info("⏳ No Accum")
                
                if detail['wick_signals']:
                    with st.expander("⚠️ Peringatan Wick Panjang (Manipulasi Whale)"):
                        for w in detail['wick_signals']:
                            st.write(f"- {w}")
                
                with st.expander("📝 Detail Analisis Lengkap"):
                    for reason in detail['reasons'][:20]:
                        st.write(f"- {reason}")
                
                # Chart
                st.plotly_chart(plot_advanced_chart(
                    detail['daily_df'], 
                    detail['symbol'],
                    detail['nearest_support'],
                    detail['nearest_resistance'],
                    detail['trendline_analysis']
                ), use_container_width=True)
                
                st.caption("⚠️ **Disclaimer:** Analisis berdasarkan data historis dan probabilitas. Bukan jaminan keuntungan. Selalu gunakan manajemen risiko!")

# Manual analysis
if manual_button and manual_symbol:
    symbol = manual_symbol.strip().upper()
    if not symbol.endswith('/USDT'):
        symbol += '/USDT'
    
    with st.spinner(f"Menganalisis {symbol}..."):
        result = analyze_coin_spot(symbol, exchange_name)
        
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
    
    if detail['final_trend'] == "BEARISH (Turun)":
        box_class = "avoid-box"
    elif detail['score'] >= 8:
        box_class = "buy-box"
    elif detail['score'] >= 6:
        box_class = "buy-box"
    elif detail['score'] >= 4:
        box_class = "wait-box"
    else:
        box_class = "avoid-box"
    
    st.markdown(f"""
    <div class="{box_class}">
        <h3>{detail['status']}</h3>
        <p><b>Skor:</b> {detail['score']}/10 | <b>Konfidensi:</b> {detail['confidence']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # VOLUME SPIKE
    if detail['has_volume_spike']:
        st.markdown(f"""
        <div class="volume-box">
            <h4>🔊 VOLUME SPIKE {detail['spike_ratio']}x!</h4>
            <p>{detail['trend_prediction']['volume_interpretation'] if detail['trend_prediction'] else 'Ada lonjakan volume signifikan!'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ICHIMOKU BOX
    if detail['ichimoku_summary']:
        ichi = detail['ichimoku_summary']
        st.markdown(f"""
        <div class="ichimoku-box">
            <h4>☁️ Ichimoku Cloud Analysis</h4>
            <table style="width: 100%;">
                <tr><td style="width: 40%;"><b>Posisi Harga:</b></td>
                <td>{ichi['cloud_position']} - {ichi['cloud_status']}</td>
                </tr>
                <tr><td><b>Ketebalan Cloud:</b></td>
                <td>{ichi['cloud_thickness_text']} ({ichi['cloud_thickness']}%)</td>
                </tr>
                <tr><td><b>TK Cross:</b></td>
                <td>{ichi['tk_status']}</td>
                </tr>
                <tr><td><b>Chikou Span:</b></td>
                <td>{ichi['chikou_status']}</td>
                </tr>
                <tr><td><b>Future Cloud:</b></td>
                <td>{ichi['future_status']}</td>
                </tr>
            </table>
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
        for ws in detail['whale_signals'][:4]:
            st.write(f"- {ws}")
    
    # TIMING
    if detail['entry_timing_status'] == "✅ BELI SEKARANG":
        st.success(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
    elif detail['entry_timing_status'] == "⬇️ TUNGGU PULLBACK":
        st.warning(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
    elif detail['entry_timing_status'] == "⚡ AGGRESSIVE ENTRY":
        st.info(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
    elif detail['entry_timing_status'] == "⏸️ HOLD DULU":
        st.error(f"⏰ **{detail['entry_timing_status']}** - {detail['entry_timing_reason']}")
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
        if detail['final_trend'] == "BULLISH (Naik)":
            st.metric("📊 Risk/Reward", f"1:{detail['rr_ratio']:.1f}")
        else:
            st.metric("📊 Risk/Reward", "N/A")
    
    # TARGET PROFIT
    st.write(f"**🎯 {detail['target_type']}:**")
    col1, col2, col3 = st.columns(3)
    
    if detail['final_trend'] == "BULLISH (Naik)":
        with col1:
            st.info(f"Target 1: ${detail['tp1']:.6f} (+{detail['tp1_pct']:.1f}%)")
        with col2:
            st.info(f"Target 2: ${detail['tp2']:.6f} (+{detail['tp2_pct']:.1f}%)")
        with col3:
            st.success(f"Target 3: ${detail['tp3']:.6f} (+{detail['tp3_pct']:.1f}%)")
    elif detail['final_trend'] == "BEARISH (Turun)":
        with col1:
            st.warning(f"Zone 1: ${detail['tp1']:.6f} ({detail['tp1_pct']:.1f}%)")
        with col2:
            st.warning(f"Zone 2: ${detail['tp2']:.6f} ({detail['tp2_pct']:.1f}%)")
        with col3:
            st.warning(f"Zone 3: ${detail['tp3']:.6f} ({detail['tp3_pct']:.1f}%)")
        st.caption("💡 Ini adalah level BELI (DCA), BUKAN take profit. Harga diprediksi turun dulu.")
    else:
        with col1:
            st.info(f"Target 1: ${detail['tp1']:.6f} ({detail['tp1_pct']:+.1f}%)")
        with col2:
            st.info(f"Target 2: ${detail['tp2']:.6f} ({detail['tp2_pct']:+.1f}%)")
        with col3:
            st.info(f"Target 3: ${detail['tp3']:.6f} ({detail['tp3_pct']:+.1f}%)")
    
    st.info(detail['beginner_summary'])
    st.write(f"**📊 Support:** ${detail['nearest_support']:.6f} | **Resistance:** ${detail['nearest_resistance']:.6f}")
    
    with st.expander("📝 Detail Analisis Lengkap"):
        for reason in detail['reasons'][:20]:
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
**🔮 FITUR SUPER LENGKAP (FINAL):**
- ✅ **Ichimoku Cloud** - Posisi Cloud, TK Cross, Chikou, Future Cloud, Ketebalan
- ✅ **Prediksi Trend 30 Hari** - 5 komponen (Ichimoku, Trend, Volume, RSI, Whale)
- ✅ **Volume Spike Detection** - Deteksi whale masuk dengan bobot tinggi (bisa override sinyal bearish!)
- ✅ **Whale Detection** - OBV Divergence, Volume Profile, Wick Manipulation
- ✅ **Garis Trend Otomatis** - Deteksi support/resistance dinamis
- ✅ **Prediksi Breakout** - Waktu dan harga breakout (tidak kontradiksi)
- ✅ **Pola Candlestick** - Prediksi 5 hari ke depan

**📌 Target Profit (SUDAH FIX):**
- ✅ **BULLISH** → Target Take Profit (harga naik) - lebih besar jika volume spike
- ✅ **BEARISH** → Target Buy Zone (harga turun untuk entry DCA)
- ✅ **SIDEWAYS + Volume Spike** → Bisa dianggap BULLISH karena volume besar

**💡 Untuk Spot Trading:**
- BULLISH + Volume Spike 3x+ → Target EXTRA (pump besar!)
- BULLISH + No Volume → Target konservatif
- BEARISH → HOLD DULU, tunggu harga turun ke buy zone
""")
