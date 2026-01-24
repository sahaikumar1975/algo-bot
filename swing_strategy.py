"""
Industry-Standard Swing Trading Strategy Module

This module implements a professional-grade swing trading strategy combining:
- SMA (Simple Moving Averages): 21, 50, 200 for trend identification
- RSI (Relative Strength Index): Momentum oscillator for overbought/oversold
- MACD (Moving Average Convergence Divergence): Trend and momentum confirmation
- ADX (Average Directional Index): Trend strength filter
- Bollinger Bands: Volatility measurement
- ATR (Average True Range): Stop-loss and risk management
- Volume Analysis: OBV and volume confirmation

Strategy Logic:
1. Trend Alignment: Price above SMA50 and SMA200, SMA50 > SMA200 (Golden alignment)
2. Pullback Entry: RSI in 40-60 zone (not overbought), price near SMA21
3. Momentum Confirmation: MACD line > Signal line, ADX > 25
4. Volume Confirmation: OBV trending up or volume above average
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def add_sma(df: pd.DataFrame, window: int, column: str = 'Close') -> pd.DataFrame:
    """Add Simple Moving Average."""
    df[f'SMA{window}'] = df[column].rolling(window=window, min_periods=1).mean()
    return df


def add_ema(df: pd.DataFrame, window: int, column: str = 'Close') -> pd.DataFrame:
    """Add Exponential Moving Average."""
    df[f'EMA{window}'] = df[column].ewm(span=window, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD (Moving Average Convergence Divergence)."""
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average Directional Index (ADX) for trend strength."""
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    n = len(df)

    # True Range
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))

        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Smoothed using Wilder's smoothing (EMA with alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    adx = np.zeros(n)

    # Initialize with simple averages
    atr[period] = np.mean(tr[1:period+1])
    smooth_plus_dm = np.mean(plus_dm[1:period+1])
    smooth_minus_dm = np.mean(minus_dm[1:period+1])

    if atr[period] > 0:
        plus_di[period] = 100 * smooth_plus_dm / atr[period]
        minus_di[period] = 100 * smooth_minus_dm / atr[period]

    # Wilder's smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        smooth_plus_dm = (smooth_plus_dm * (period - 1) + plus_dm[i]) / period
        smooth_minus_dm = (smooth_minus_dm * (period - 1) + minus_dm[i]) / period

        if atr[i] > 0:
            plus_di[i] = 100 * smooth_plus_dm / atr[i]
            minus_di[i] = 100 * smooth_minus_dm / atr[i]

    # DX and ADX
    dx = np.zeros(n)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])

    # ADX is smoothed DX
    adx[2*period] = np.mean(dx[period:2*period+1])
    for i in range(2*period + 1, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    df['ADX'] = adx
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di

    # Replace zeros with NaN for cleaner display
    df.loc[df['ADX'] == 0, 'ADX'] = np.nan

    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands."""
    sma = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = sma + (std_dev * std)
    df['BB_Middle'] = sma
    df['BB_Lower'] = sma - (std_dev * std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
    df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']) * 100
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range (ATR) for volatility and stop-loss calculation."""
    high = df['High']
    low = df['Low']
    close = df['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df['ATR'] = tr.rolling(window=period).mean()
    df['ATR_Percent'] = df['ATR'] / df['Close'] * 100
    return df


def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """Add On-Balance Volume (OBV)."""
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_SMA20'] = df['OBV'].rolling(window=20).mean()
    return df


def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Add Stochastic Oscillator."""
    low_min = df['Low'].rolling(window=k_period).min()
    high_max = df['High'].rolling(window=k_period).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to the dataframe."""
    df = add_sma(df, 21)
    df = add_sma(df, 50)
    df = add_sma(df, 200)
    df = add_ema(df, 9)
    df = add_ema(df, 21)
    df = add_rsi(df, 14)
    df = add_macd(df)
    df = add_adx(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_stochastic(df)

    # Volume analysis
    df['Vol_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_SMA20']

    return df


# =============================================================================
# SWING TRADING STRATEGY - PROFESSIONAL GRADE
# =============================================================================

def calculate_signal_strength(df: pd.DataFrame) -> Dict:
    """
    Calculate a composite signal strength score (0-100).

    Scoring Components:
    - Trend Alignment (25 points): SMA stacking (21>50>200)
    - RSI Position (20 points): Ideal zone 40-60
    - MACD Momentum (20 points): MACD > Signal and histogram positive
    - ADX Trend Strength (15 points): ADX > 25
    - Volume Confirmation (10 points): Volume above average
    - Bollinger Position (10 points): Not overbought
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    breakdown = {}

    # 1. Trend Alignment (25 points)
    trend_score = 0
    if last['Close'] > last['SMA21']:
        trend_score += 5
    if last['Close'] > last['SMA50']:
        trend_score += 5
    if last['Close'] > last['SMA200']:
        trend_score += 5
    if last['SMA21'] > last['SMA50']:
        trend_score += 5
    if last['SMA50'] > last['SMA200']:
        trend_score += 5
    breakdown['trend'] = trend_score
    score += trend_score

    # 2. RSI Position (20 points)
    rsi = last['RSI']
    rsi_score = 0
    if not pd.isna(rsi):
        if 40 <= rsi <= 60:  # Ideal pullback zone
            rsi_score = 20
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            rsi_score = 15
        elif 50 <= rsi <= 70:  # Bullish but not overbought
            rsi_score = 12
        elif rsi < 30:  # Oversold - could be value
            rsi_score = 10
        elif rsi > 70:  # Overbought - risky
            rsi_score = 5
    breakdown['rsi'] = rsi_score
    score += rsi_score

    # 3. MACD Momentum (20 points)
    macd_score = 0
    if not pd.isna(last['MACD']) and not pd.isna(last['MACD_Signal']):
        if last['MACD'] > last['MACD_Signal']:
            macd_score += 10
        if last['MACD_Hist'] > 0:
            macd_score += 5
        if last['MACD_Hist'] > prev['MACD_Hist']:  # Increasing momentum
            macd_score += 5
    breakdown['macd'] = macd_score
    score += macd_score

    # 4. ADX Trend Strength (15 points)
    adx_score = 0
    adx = last['ADX']
    if not pd.isna(adx):
        if adx >= 40:  # Very strong trend
            adx_score = 15
        elif adx >= 30:  # Strong trend
            adx_score = 12
        elif adx >= 25:  # Trending
            adx_score = 10
        elif adx >= 20:  # Weak trend
            adx_score = 5
    breakdown['adx'] = adx_score
    score += adx_score

    # 5. Volume Confirmation (10 points)
    vol_score = 0
    vol_ratio = last['Vol_Ratio']
    if not pd.isna(vol_ratio):
        if vol_ratio >= 1.5:
            vol_score = 10
        elif vol_ratio >= 1.2:
            vol_score = 7
        elif vol_ratio >= 1.0:
            vol_score = 5
    breakdown['volume'] = vol_score
    score += vol_score

    # 6. Bollinger Position (10 points)
    bb_score = 0
    bb_pct = last['BB_Percent']
    if not pd.isna(bb_pct):
        if 30 <= bb_pct <= 70:  # Middle of bands - healthy
            bb_score = 10
        elif 20 <= bb_pct < 30 or 70 < bb_pct <= 80:
            bb_score = 7
        elif bb_pct < 20:  # Near lower band - potential bounce
            bb_score = 8
        elif bb_pct > 80:  # Near upper band - extended
            bb_score = 3
    breakdown['bollinger'] = bb_score
    score += bb_score

    return {
        'total_score': score,
        'grade': _score_to_grade(score),
        'breakdown': breakdown
    }


def _score_to_grade(score: int) -> str:
    """Convert numerical score to letter grade."""
    if score >= 85:
        return 'A+'
    elif score >= 75:
        return 'A'
    elif score >= 65:
        return 'B+'
    elif score >= 55:
        return 'B'
    elif score >= 45:
        return 'C+'
    elif score >= 35:
        return 'C'
    else:
        return 'D'


def detect_swing_signal(df: pd.DataFrame, min_score: int = 50) -> Dict:
    """
    Professional Swing Trading Signal Detection.

    Entry Criteria:
    1. Trend: Price > SMA50 > SMA200 (uptrend confirmed)
    2. Pullback: Price near SMA21 (within 3%) or just crossed above
    3. RSI: Between 30-65 (not overbought)
    4. MACD: MACD line > Signal line OR histogram turning positive
    5. ADX: > 20 (trending market)
    6. Volume: Recent volume spike or OBV trending up

    Returns signal with entry, stop-loss, and targets.
    """
    out = {'signal': False, 'signal_type': 'swing_professional'}

    if df is None or df.shape[0] < 50:
        return out

    df = df.copy()
    df = add_all_indicators(df)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Basic data extraction
    close = last['Close']
    sma21 = last['SMA21']
    sma50 = last['SMA50']
    sma200 = last['SMA200']
    rsi = last['RSI']
    macd = last['MACD']
    macd_signal = last['MACD_Signal']
    adx = last['ADX']
    atr = last['ATR']

    # Check for NaN values (ADX can be NaN for first ~28 periods, which is OK)
    if any(pd.isna([sma21, sma50, sma200, rsi, macd, atr])):
        return out

    # =================================
    # ENTRY CONDITIONS
    # =================================

    # 1. Trend Alignment (Required)
    uptrend = close > sma50 and sma50 > sma200

    # 2. Pullback Condition (Price near SMA21)
    dist_to_sma21 = (close - sma21) / sma21 * 100  # Positive if above, negative if below
    near_sma21 = -2.0 <= dist_to_sma21 <= 5.0  # Allow 2% below to 5% above SMA21
    just_crossed_sma21 = prev['Close'] <= prev['SMA21'] and close > sma21
    pullback_entry = near_sma21 or just_crossed_sma21

    # 3. RSI Filter (Not overbought, expanded range)
    rsi_ok = 30 <= rsi <= 70

    # 4. MACD Confirmation (bullish, improving, or histogram not too negative)
    macd_bullish = (macd > macd_signal or
                    last['MACD_Hist'] > prev['MACD_Hist'] or
                    last['MACD_Hist'] > -abs(macd) * 0.5)  # Hist not deeply negative

    # 5. ADX Filter (Trending market) - skip if ADX not available
    adx_ok = pd.isna(adx) or adx >= 18  # Relaxed from 20, allow NaN

    # 6. SMA200 Ascending (Long-term uptrend)
    sma200_ascending = last['SMA200'] > prev['SMA200']

    # Combined signal - core requirements
    signal = uptrend and pullback_entry and rsi_ok and macd_bullish and adx_ok and sma200_ascending

    if signal:
        # Calculate signal strength
        strength = calculate_signal_strength(df)

        # Only generate signal if score meets minimum
        if strength['total_score'] < min_score:
            return out

        # Calculate stop-loss and targets
        stop_loss = close - (2 * atr)  # 2 ATR below entry
        target1 = close + (2 * atr)     # 1:1 R:R
        target2 = close + (3 * atr)     # 1.5:1 R:R
        target3 = close + (4 * atr)     # 2:1 R:R

        # Risk calculation
        risk_per_share = close - stop_loss
        risk_percent = risk_per_share / close * 100

        out['signal'] = True
        out['date'] = last.name
        out['close'] = float(close)
        out['sma21'] = float(sma21)
        out['sma50'] = float(sma50)
        out['sma200'] = float(sma200)
        out['rsi'] = float(rsi)
        out['macd'] = float(macd)
        out['macd_signal'] = float(macd_signal)
        out['adx'] = float(adx) if not pd.isna(adx) else 0.0
        out['atr'] = float(atr)
        out['stop_loss'] = float(stop_loss)
        out['target1'] = float(target1)
        out['target2'] = float(target2)
        out['target3'] = float(target3)
        out['risk_percent'] = float(risk_percent)
        out['signal_strength'] = strength['total_score']
        out['signal_grade'] = strength['grade']
        out['strength_breakdown'] = strength['breakdown']
        out['dist_to_sma21'] = float(dist_to_sma21)

    return out


def fetch_ohlc_extended(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLC data with extended history for backtesting."""
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval)
    if df.empty:
        return df
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    return df


def screen_swing_signals(tickers: List[str], period: str = '1y',
                         min_score: int = 50, include_df: bool = False) -> List[Dict]:
    """
    Screen tickers for professional swing trading signals.

    Args:
        tickers: List of ticker symbols
        period: Historical period for analysis
        min_score: Minimum signal strength score (0-100)
        include_df: Include full DataFrame with indicators

    Returns:
        List of signal dictionaries sorted by strength
    """
    results = []

    for ticker in tickers:
        try:
            df = fetch_ohlc_extended(ticker, period=period)
            if df.empty or len(df) < 50:
                continue

            signal = detect_swing_signal(df, min_score=min_score)

            if signal.get('signal'):
                result = {
                    'ticker': ticker,
                    **signal
                }
                if include_df:
                    df = add_all_indicators(df)
                    result['ohlc_df'] = df
                results.append(result)

        except Exception as e:
            continue

    # Sort by signal strength (highest first)
    results.sort(key=lambda x: x.get('signal_strength', 0), reverse=True)

    return results


# =============================================================================
# WEEKLY HIGH/LOW BREAKOUT SCREENER
# =============================================================================

def fetch_weekly_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch weekly OHLC data for a ticker."""
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval="1wk")
    if df.empty:
        return df
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    return df


def detect_weekly_breakout_pullback(ticker: str) -> Dict:
    """
    Detect stocks that crossed weekly high but closed below it (pullback setup).

    Strategy Logic:
    1. Get the previous week's high and low from weekly timeframe
    2. Check daily timeframe if price crossed above weekly high during the week
    3. Check if daily close is now below the weekly high (pullback)

    This identifies potential breakout-pullback entries where:
    - Stock showed strength by crossing weekly high
    - But pulled back, offering a better entry point

    Returns signal dict with weekly levels and daily price action.
    """
    out = {
        'signal': False,
        'signal_type': 'weekly_breakout_pullback',
        'ticker': ticker.replace('.NS', '')
    }

    try:
        # Fetch weekly data (last 4 weeks)
        weekly_df = fetch_weekly_data(ticker, period="1mo")
        if weekly_df.empty or len(weekly_df) < 2:
            return out

        # Get previous week's high and low (second last completed week)
        prev_week = weekly_df.iloc[-2]
        weekly_high = prev_week['High']
        weekly_low = prev_week['Low']

        # Fetch daily data (last 10 days)
        daily_df = fetch_ohlc_extended(ticker, period="1mo", interval="1d")
        if daily_df.empty or len(daily_df) < 5:
            return out

        # Get current week's daily data (last 5 trading days approximately)
        recent_daily = daily_df.tail(7)

        # Check if any day in current week crossed above weekly high
        crossed_weekly_high = any(recent_daily['High'] > weekly_high)

        # Get latest daily close
        last_daily = daily_df.iloc[-1]
        current_close = last_daily['Close']
        current_high = last_daily['High']

        # Check if current close is below weekly high (pullback condition)
        closed_below_weekly_high = current_close < weekly_high

        # Additional: Price should still be above weekly low (not breakdown)
        above_weekly_low = current_close > weekly_low

        # Signal: Crossed weekly high but closed below it (pullback)
        if crossed_weekly_high and closed_below_weekly_high and above_weekly_low:
            out['signal'] = True
            out['weekly_high'] = float(weekly_high)
            out['weekly_low'] = float(weekly_low)
            out['current_close'] = float(current_close)
            out['current_high'] = float(current_high)
            out['distance_from_weekly_high'] = float((weekly_high - current_close) / weekly_high * 100)
            out['distance_from_weekly_low'] = float((current_close - weekly_low) / weekly_low * 100)
            out['weekly_range'] = float(weekly_high - weekly_low)
            out['weekly_range_pct'] = float((weekly_high - weekly_low) / weekly_low * 100)

            # Determine setup quality
            dist_from_high = out['distance_from_weekly_high']
            if dist_from_high <= 1.0:
                out['setup_quality'] = 'A - Near Breakout'
            elif dist_from_high <= 2.5:
                out['setup_quality'] = 'B - Good Pullback'
            elif dist_from_high <= 5.0:
                out['setup_quality'] = 'C - Deep Pullback'
            else:
                out['setup_quality'] = 'D - Too Far'

            # Suggested levels
            out['entry_zone'] = f"₹{current_close:.2f} - ₹{weekly_high:.2f}"
            out['stop_loss'] = float(weekly_low - (weekly_high - weekly_low) * 0.1)  # Below weekly low
            out['target1'] = float(weekly_high + (weekly_high - weekly_low) * 0.5)  # 50% extension
            out['target2'] = float(weekly_high + (weekly_high - weekly_low))  # 100% extension

    except Exception as e:
        out['error'] = str(e)

    return out


def screen_weekly_breakout(tickers: List[str]) -> List[Dict]:
    """
    Screen all tickers for weekly high breakout-pullback setups.

    Returns list of stocks that:
    1. Crossed their previous weekly high
    2. But closed below the weekly high (offering pullback entry)
    """
    results = []

    for ticker in tickers:
        try:
            signal = detect_weekly_breakout_pullback(ticker)
            if signal.get('signal'):
                results.append(signal)
        except Exception:
            continue

    # Sort by distance from weekly high (closest first = best setup)
    results.sort(key=lambda x: x.get('distance_from_weekly_high', 999))

    return results


def screen_52week_high(tickers: List[str], tolerance_pct: float = 2.0) -> List[Dict]:
    """
    Screen all tickers near their 52-week highs (within tolerance_pct).

    Args:
        tickers: List of ticker symbols
        tolerance_pct: Percentage tolerance from 52-week high (default 2%)

    Returns:
        List of stocks trading near 52-week highs, sorted by distance
    """
    results = []

    for ticker in tickers:
        try:
            # Fetch 1 year of data
            df = fetch_ohlc_extended(ticker, period='1y')
            if df.empty or len(df) < 50:
                continue

            current_close = df['Close'].iloc[-1]
            high_52week = df['High'].max()
            
            # Calculate distance from 52-week high
            distance_pct = ((high_52week - current_close) / high_52week) * 100
            
            # If within tolerance range
            if distance_pct <= tolerance_pct:
                # Convert ticker format from yfinance (RELIANCE.NS) to NSE format (NSE:RELIANCE-EQ)
                display_ticker = ticker.replace('.NS', '')
                fyers_ticker = f"NSE:{display_ticker}-EQ"
                
                results.append({
                    'ticker': fyers_ticker,
                    'current_price': round(current_close, 2),
                    '52week_high': round(high_52week, 2),
                    'distance_from_high_pct': round(distance_pct, 2),
                    'date': df.index[-1].strftime('%Y-%m-%d'),
                    'strength': 'STRONG' if distance_pct <= 0.5 else 'HIGH' if distance_pct <= 1.0 else 'GOOD'
                })
        except Exception:
            continue

    # Sort by distance from 52-week high (closest = strongest signal)
    results.sort(key=lambda x: x.get('distance_from_high_pct', 999))

    return results


def screen_high_volume_trend(tickers: List[str], lookback_days: int = 3) -> List[Dict]:
    """
    Screen all tickers with highest volume traded for continuous days.

    Args:
        tickers: List of ticker symbols
        lookback_days: Number of consecutive days to check (default 3)

    Returns:
        List of stocks with high volume trends, sorted by volume momentum
    """
    results = []

    for ticker in tickers:
        try:
            # Fetch 1 month of data to have enough history
            df = fetch_ohlc_extended(ticker, period='1mo')
            if df.empty or len(df) < lookback_days:
                continue

            # Get last N days
            last_n_days = df.tail(lookback_days).copy()
            volumes = last_n_days['Volume'].values
            
            # Check if volumes are continuously high (last volume is highest or near highest in period)
            avg_volume_all = df['Volume'].mean()
            recent_avg_volume = volumes.mean()
            
            # Check for continuous high volume (each day >= 90% of highest in the lookback period)
            max_volume = volumes.max()
            min_volume_threshold = max_volume * 0.85  # 85% of max
            
            is_continuous_high = all(v >= min_volume_threshold for v in volumes)
            
            if is_continuous_high and recent_avg_volume > avg_volume_all:
                current_price = df['Close'].iloc[-1]
                current_volume = df['Volume'].iloc[-1]
                
                # Calculate volume momentum
                volume_momentum = ((recent_avg_volume - avg_volume_all) / avg_volume_all) * 100
                
                # Get price change over the period
                price_change_pct = ((current_price - last_n_days['Close'].iloc[0]) / last_n_days['Close'].iloc[0]) * 100
                
                # Convert ticker format
                display_ticker = ticker.replace('.NS', '')
                fyers_ticker = f"NSE:{display_ticker}-EQ"
                
                results.append({
                    'ticker': fyers_ticker,
                    'current_price': round(current_price, 2),
                    'current_volume': f"{current_volume:,.0f}",
                    'avg_volume_3d': f"{recent_avg_volume:,.0f}",
                    'avg_volume_all': f"{avg_volume_all:,.0f}",
                    'volume_momentum_pct': round(volume_momentum, 2),
                    'price_change_pct': round(price_change_pct, 2),
                    'date': df.index[-1].strftime('%Y-%m-%d'),
                    'strength': 'EXTREME' if volume_momentum > 100 else 'VERY_HIGH' if volume_momentum > 50 else 'HIGH'
                })
        except Exception:
            continue

    # Sort by volume momentum (highest first)
    results.sort(key=lambda x: x.get('volume_momentum_pct', 0), reverse=True)

    return results


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

def backtest_strategy(ticker: str, period: str = '2y',
                      initial_capital: float = 100000,
                      risk_per_trade: float = 0.02,
                      min_score: int = 50) -> Dict:
    """
    Backtest the swing trading strategy on historical data.

    Args:
        ticker: Stock ticker symbol
        period: Historical period (e.g., '2y', '5y')
        initial_capital: Starting capital
        risk_per_trade: Risk per trade as fraction (0.02 = 2%)
        min_score: Minimum signal strength for entry

    Returns:
        Dictionary with backtest results and trade history
    """
    df = fetch_ohlc_extended(ticker, period=period)
    if df.empty or len(df) < 200:
        return {'error': 'Insufficient data'}

    df = add_all_indicators(df)

    # Initialize backtest variables
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_date = None
    stop_loss = 0
    target1 = 0
    target2 = 0
    target3 = 0

    trades = []
    equity_curve = []

    # Skip first 200 days to have enough indicator data
    for i in range(200, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        date = df.index[i]

        # Track equity
        if position > 0:
            current_value = capital + position * current['Close']
        else:
            current_value = capital
        equity_curve.append({'date': date, 'equity': current_value})

        # If in position, check exit conditions
        if position > 0:
            # Stop-loss hit
            if current['Low'] <= stop_loss:
                exit_price = stop_loss
                pnl = (exit_price - entry_price) * position
                capital += position * exit_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': position,
                    'pnl': pnl,
                    'pnl_percent': (exit_price / entry_price - 1) * 100,
                    'exit_reason': 'stop_loss'
                })
                position = 0
                continue

            # Target 2 hit (take full profit)
            if current['High'] >= target2:
                exit_price = target2
                pnl = (exit_price - entry_price) * position
                capital += position * exit_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': position,
                    'pnl': pnl,
                    'pnl_percent': (exit_price / entry_price - 1) * 100,
                    'exit_reason': 'target_hit'
                })
                position = 0
                continue

            # Exit if RSI > 75 (overbought)
            if current['RSI'] > 75:
                exit_price = current['Close']
                pnl = (exit_price - entry_price) * position
                capital += position * exit_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': position,
                    'pnl': pnl,
                    'pnl_percent': (exit_price / entry_price - 1) * 100,
                    'exit_reason': 'rsi_overbought'
                })
                position = 0
                continue

            # Exit if price closes below SMA50
            if current['Close'] < current['SMA50']:
                exit_price = current['Close']
                pnl = (exit_price - entry_price) * position
                capital += position * exit_price
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'shares': position,
                    'pnl': pnl,
                    'pnl_percent': (exit_price / entry_price - 1) * 100,
                    'exit_reason': 'sma50_break'
                })
                position = 0
                continue

        # If no position, check entry conditions
        if position == 0:
            # Create a slice for signal detection
            df_slice = df.iloc[:i+1].copy()
            signal = detect_swing_signal(df_slice, min_score=min_score)

            if signal.get('signal'):
                # Position sizing based on risk
                atr = current['ATR']
                risk_amount = capital * risk_per_trade
                stop_distance = 2 * atr
                shares = int(risk_amount / stop_distance)

                if shares > 0 and shares * current['Close'] <= capital:
                    entry_price = current['Close']
                    entry_date = date
                    stop_loss = entry_price - stop_distance
                    target1 = entry_price + (2 * atr)
                    target2 = entry_price + (3 * atr)
                    target3 = entry_price + (4 * atr)
                    position = shares
                    capital -= position * entry_price

    # Close any open position at the end
    if position > 0:
        exit_price = df.iloc[-1]['Close']
        pnl = (exit_price - entry_price) * position
        capital += position * exit_price
        trades.append({
            'entry_date': entry_date,
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': position,
            'pnl': pnl,
            'pnl_percent': (exit_price / entry_price - 1) * 100,
            'exit_reason': 'end_of_period'
        })

    # Calculate metrics
    if not trades:
        return {
            'ticker': ticker,
            'total_trades': 0,
            'message': 'No trades generated'
        }

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)

    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]

    total_pnl = trades_df['pnl'].sum()
    final_capital = initial_capital + total_pnl

    # Calculate max drawdown
    equity_df['peak'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak'] * 100
    max_drawdown = equity_df['drawdown'].min()

    # Win rate and profit factor
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Average trade stats
    avg_win = winning_trades['pnl_percent'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_percent'].mean() if len(losing_trades) > 0 else 0
    avg_trade = trades_df['pnl_percent'].mean()

    # Holding period
    trades_df['holding_days'] = (pd.to_datetime(trades_df['exit_date']) -
                                  pd.to_datetime(trades_df['entry_date'])).dt.days
    avg_holding = trades_df['holding_days'].mean()

    return {
        'ticker': ticker,
        'period': period,
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return_pct': (final_capital / initial_capital - 1) * 100,
        'total_pnl': total_pnl,
        'total_trades': len(trades_df),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_drawdown,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'avg_trade_pct': avg_trade,
        'avg_holding_days': avg_holding,
        'trades': trades,
        'equity_curve': equity_curve
    }


def calculate_risk_metrics(account_size: float, entry_price: float,
                           stop_loss: float, risk_percent: float = 0.02) -> Dict:
    """
    Calculate position sizing and risk metrics.

    Args:
        account_size: Total account capital
        entry_price: Planned entry price
        stop_loss: Stop-loss price
        risk_percent: Max risk per trade (default 2%)

    Returns:
        Dictionary with position size and risk calculations
    """
    risk_amount = account_size * risk_percent
    risk_per_share = entry_price - stop_loss

    if risk_per_share <= 0:
        return {'error': 'Invalid stop-loss (must be below entry)'}

    position_size = int(risk_amount / risk_per_share)
    position_value = position_size * entry_price

    return {
        'account_size': account_size,
        'risk_percent': risk_percent * 100,
        'risk_amount': risk_amount,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'risk_per_share': risk_per_share,
        'position_size': position_size,
        'position_value': position_value,
        'position_percent': position_value / account_size * 100
    }


# =============================================================================
# SECTOR MAPPING AND ANALYSIS
# =============================================================================

# Comprehensive sector mapping for Nifty 200 stocks
SECTOR_MAPPING = {
    # Banking & Financial Services
    'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'SBIN': 'Banking', 'KOTAKBANK': 'Banking',
    'AXISBANK': 'Banking', 'INDUSINDBK': 'Banking', 'BANKBARODA': 'Banking', 'PNB': 'Banking',
    'CANBK': 'Banking', 'IDFCFIRSTB': 'Banking', 'FEDERALBNK': 'Banking', 'AUBANK': 'Banking',
    'BANDHANBNK': 'Banking', 'RBLBANK': 'Banking', 'CUB': 'Banking', 'KARURVYSYA': 'Banking',
    'IDFC': 'Banking',

    # NBFC & Financial Services
    'BAJFINANCE': 'NBFC', 'BAJAJFINSV': 'NBFC', 'CHOLAFIN': 'NBFC', 'M&MFIN': 'NBFC',
    'SHRIRAMFIN': 'NBFC', 'MUTHOOTFIN': 'NBFC', 'MANAPPURAM': 'NBFC', 'AAVAS': 'NBFC',
    'CANFINHOME': 'NBFC', 'LICHSGFIN': 'NBFC', 'SUNDARMFIN': 'NBFC', 'POONAWALLA': 'NBFC',
    'ABCAPITAL': 'NBFC', 'PFC': 'NBFC', 'RECLTD': 'NBFC', 'BAJAJHLDNG': 'NBFC',

    # Insurance & AMC
    'SBILIFE': 'Insurance', 'HDFCLIFE': 'Insurance', 'ICICIGI': 'Insurance', 'ICICIPRULI': 'Insurance',
    'HDFCAMC': 'AMC', 'SBICARD': 'Financial Services', 'CAMS': 'Financial Services', 'CDSL': 'Financial Services',

    # IT & Technology
    'TCS': 'IT', 'INFY': 'IT', 'HCLTECH': 'IT', 'WIPRO': 'IT', 'TECHM': 'IT',
    'LTIM': 'IT', 'MPHASIS': 'IT', 'COFORGE': 'IT', 'PERSISTENT': 'IT', 'LTTS': 'IT',
    'BIRLASOFT': 'IT', 'MINDTREE': 'IT', 'CYIENT': 'IT', 'KPITTECH': 'IT', 'TATAELXSI': 'IT',
    'NAUKRI': 'IT', 'OFSS': 'IT',

    # Telecom & Media
    'BHARTIARTL': 'Telecom', 'TATACOMM': 'Telecom', 'IRCTC': 'IT Services',
    'ZEEL': 'Media', 'PVR': 'Media', 'SUNTV': 'Media', 'STAR': 'Media',

    # Pharma & Healthcare
    'SUNPHARMA': 'Pharma', 'DRREDDY': 'Pharma', 'CIPLA': 'Pharma', 'DIVISLAB': 'Pharma',
    'TORNTPHARM': 'Pharma', 'LUPIN': 'Pharma', 'AUROPHARMA': 'Pharma', 'BIOCON': 'Pharma',
    'ALKEM': 'Pharma', 'APOLLOHOSP': 'Healthcare', 'MAXHEALTH': 'Healthcare', 'FORTIS': 'Healthcare',
    'LALPATHLAB': 'Healthcare',

    # Auto & Auto Ancillary
    'MARUTI': 'Auto', 'TATAMOTORS': 'Auto', 'M&M': 'Auto', 'BAJAJ-AUTO': 'Auto',
    'HEROMOTOCO': 'Auto', 'EICHERMOT': 'Auto', 'TVSMOTOR': 'Auto', 'ASHOKLEY': 'Auto',
    'BHARATFORG': 'Auto Ancillary', 'BALKRISIND': 'Auto Ancillary', 'MRF': 'Auto Ancillary',
    'APOLLOTYRE': 'Auto Ancillary', 'BOSCHLTD': 'Auto Ancillary', 'MOTHERSON': 'Auto Ancillary',
    'EXIDEIND': 'Auto Ancillary', 'SONACOMS': 'Auto Ancillary', 'ENDURANCE': 'Auto Ancillary',
    'ESCORTS': 'Auto Ancillary',

    # Oil & Gas
    'RELIANCE': 'Oil & Gas', 'ONGC': 'Oil & Gas', 'BPCL': 'Oil & Gas', 'IOC': 'Oil & Gas',
    'GAIL': 'Oil & Gas', 'ATGL': 'Oil & Gas',

    # Metals & Mining
    'TATASTEEL': 'Metals', 'JSWSTEEL': 'Metals', 'HINDALCO': 'Metals', 'VEDL': 'Metals',
    'COALINDIA': 'Mining', 'NMDC': 'Mining', 'NATIONALUM': 'Metals', 'SAIL': 'Metals',
    'JINDALSTEL': 'Metals',

    # Power & Utilities
    'NTPC': 'Power', 'POWERGRID': 'Power', 'TATAPOWER': 'Power', 'ADANIGREEN': 'Power',
    'ADANITRANS': 'Power', 'JSWENERGY': 'Power', 'NHPC': 'Power', 'SJVN': 'Power',

    # Cement & Building Materials
    'ULTRACEMCO': 'Cement', 'AMBUJACEM': 'Cement', 'ACC': 'Cement', 'SHREECEM': 'Cement',
    'DALBHARAT': 'Cement', 'RAMCOCEM': 'Cement', 'GRASIM': 'Cement',

    # FMCG
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'NESTLEIND': 'FMCG', 'BRITANNIA': 'FMCG',
    'DABUR': 'FMCG', 'GODREJCP': 'FMCG', 'MARICO': 'FMCG', 'COLPAL': 'FMCG',
    'PGHH': 'FMCG', 'TATACONSUM': 'FMCG', 'VBL': 'FMCG', 'UNITDSPR': 'FMCG', 'RADICO': 'FMCG',
    'MCDOWELL-N': 'FMCG', 'JUBLFOOD': 'FMCG',

    # Consumer Durables
    'TITAN': 'Consumer Durables', 'HAVELLS': 'Consumer Durables', 'VOLTAS': 'Consumer Durables',
    'WHIRLPOOL': 'Consumer Durables', 'CROMPTON': 'Consumer Durables', 'PAGEIND': 'Consumer Durables',
    'RELAXO': 'Consumer Durables', 'BATAINDIA': 'Consumer Durables',

    # Paints
    'ASIANPAINT': 'Paints', 'BERGEPAINT': 'Paints', 'PIDILITIND': 'Chemicals',

    # Real Estate
    'DLF': 'Real Estate', 'GODREJPROP': 'Real Estate', 'OBEROIRLTY': 'Real Estate',
    'PRESTIGE': 'Real Estate', 'BRIGADE': 'Real Estate', 'SOBHA': 'Real Estate',
    'PHOENIXLTD': 'Real Estate',

    # Retail & E-commerce
    'TRENT': 'Retail', 'DMART': 'Retail', 'ZOMATO': 'E-commerce', 'NYKAA': 'E-commerce', 'PAYTM': 'Fintech',

    # Capital Goods & Infrastructure
    'LT': 'Capital Goods', 'SIEMENS': 'Capital Goods', 'ABB': 'Capital Goods',
    'CUMMINSIND': 'Capital Goods', 'THERMAX': 'Capital Goods', 'BHEL': 'Capital Goods',
    'BEL': 'Defence', 'HAL': 'Defence', 'CGPOWER': 'Capital Goods',

    # Cables & Electricals
    'POLYCAB': 'Electricals', 'KEI': 'Electricals',

    # Chemicals
    'UPL': 'Chemicals', 'PIIND': 'Chemicals', 'SRF': 'Chemicals', 'ATUL': 'Chemicals',
    'DEEPAKNTR': 'Chemicals', 'NAVINFLUOR': 'Chemicals', 'FLUOROCHEM': 'Chemicals',
    'CLEAN': 'Chemicals', 'COROMANDEL': 'Fertilizers', 'GNFC': 'Fertilizers', 'CHAMBLFERT': 'Fertilizers',

    # Pipes & Plastics
    'ASTRAL': 'Pipes', 'SUPREMEIND': 'Pipes',

    # Aviation & Logistics
    'INDIGO': 'Aviation', 'CONCOR': 'Logistics',

    # Adani Group
    'ADANIENT': 'Conglomerate', 'ADANIPORTS': 'Ports', 'ADANIENSOL': 'Power', 'AWL': 'FMCG',
}


def get_sector(ticker: str) -> str:
    """Get sector for a given ticker symbol."""
    clean_ticker = ticker.replace('.NS', '').upper()
    return SECTOR_MAPPING.get(clean_ticker, 'Others')


def get_all_sectors() -> List[str]:
    """Get list of all unique sectors."""
    return sorted(list(set(SECTOR_MAPPING.values())))


def get_stocks_by_sector(sector: str) -> List[str]:
    """Get all stocks belonging to a sector."""
    return [ticker for ticker, sec in SECTOR_MAPPING.items() if sec == sector]


def calculate_stock_returns(ticker: str, period: str = '3mo') -> Dict:
    """
    Calculate returns for a single stock over different periods.

    Returns dict with:
    - 1 week, 1 month, 3 month returns
    - Current price and volume info
    """
    try:
        df = fetch_ohlc_extended(f"{ticker}.NS" if not ticker.endswith('.NS') else ticker, period=period)
        if df.empty or len(df) < 5:
            return {'error': 'Insufficient data'}

        current_close = df.iloc[-1]['Close']

        # Calculate returns
        returns = {
            'ticker': ticker.replace('.NS', ''),
            'sector': get_sector(ticker),
            'current_price': current_close,
        }

        # 1 week return
        if len(df) >= 5:
            returns['1w_return'] = (current_close / df.iloc[-5]['Close'] - 1) * 100
        else:
            returns['1w_return'] = 0

        # 1 month return (~21 trading days)
        if len(df) >= 21:
            returns['1m_return'] = (current_close / df.iloc[-21]['Close'] - 1) * 100
        else:
            returns['1m_return'] = 0

        # 3 month return (~63 trading days)
        if len(df) >= 63:
            returns['3m_return'] = (current_close / df.iloc[-63]['Close'] - 1) * 100
        elif len(df) > 21:
            returns['3m_return'] = (current_close / df.iloc[0]['Close'] - 1) * 100
        else:
            returns['3m_return'] = 0

        return returns
    except Exception as e:
        return {'error': str(e), 'ticker': ticker.replace('.NS', '')}


def calculate_sector_performance(tickers: List[str], period: str = '3mo') -> Dict:
    """
    Calculate sector-wise performance for a list of tickers.

    Returns:
    - Sector returns (aggregated)
    - Individual stock returns per sector
    - Sector rankings
    """
    sector_data = {}
    stock_returns = []

    for ticker in tickers:
        returns = calculate_stock_returns(ticker, period)
        if 'error' not in returns:
            stock_returns.append(returns)

            sector = returns['sector']
            if sector not in sector_data:
                sector_data[sector] = {
                    'stocks': [],
                    '1w_returns': [],
                    '1m_returns': [],
                    '3m_returns': []
                }

            sector_data[sector]['stocks'].append(returns['ticker'])
            sector_data[sector]['1w_returns'].append(returns['1w_return'])
            sector_data[sector]['1m_returns'].append(returns['1m_return'])
            sector_data[sector]['3m_returns'].append(returns['3m_return'])

    # Calculate sector averages
    sector_performance = []
    for sector, data in sector_data.items():
        if len(data['stocks']) > 0:
            sector_performance.append({
                'sector': sector,
                'stock_count': len(data['stocks']),
                'avg_1w_return': np.mean(data['1w_returns']),
                'avg_1m_return': np.mean(data['1m_returns']),
                'avg_3m_return': np.mean(data['3m_returns']),
                'best_1w': max(data['1w_returns']),
                'best_3m': max(data['3m_returns']),
                'stocks': data['stocks']
            })

    # Sort by 3-month return
    sector_performance.sort(key=lambda x: x['avg_3m_return'], reverse=True)

    return {
        'sector_performance': sector_performance,
        'stock_returns': stock_returns
    }


def get_sector_momentum_score(sector_data: Dict) -> str:
    """
    Calculate momentum score for a sector based on returns.

    Returns:
    - 'Strong Bullish' if 1w, 1m, 3m all positive and increasing
    - 'Bullish' if mostly positive
    - 'Neutral' if mixed
    - 'Bearish' if mostly negative
    - 'Strong Bearish' if all negative
    """
    avg_1w = sector_data.get('avg_1w_return', 0)
    avg_1m = sector_data.get('avg_1m_return', 0)
    avg_3m = sector_data.get('avg_3m_return', 0)

    # Count positive periods
    positive_count = sum([1 for x in [avg_1w, avg_1m, avg_3m] if x > 0])

    # Check momentum acceleration
    accelerating = avg_1w > avg_1m / 4 if avg_1m != 0 else False  # Weekly outperforming monthly pace

    if positive_count == 3 and avg_3m > 10:
        if accelerating:
            return 'Strong Bullish 🚀'
        return 'Bullish 📈'
    elif positive_count >= 2:
        return 'Bullish 📈'
    elif positive_count == 1:
        return 'Neutral ➡️'
    elif avg_3m < -10:
        return 'Strong Bearish 📉'
    else:
        return 'Bearish 📉'


def get_top_stocks_by_sector(stock_returns: List[Dict], sector: str,
                             top_n: int = 5, period: str = '3m') -> List[Dict]:
    """Get top performing stocks in a sector by return period."""
    sector_stocks = [s for s in stock_returns if s.get('sector') == sector]

    period_key = f'{period}_return'
    sector_stocks.sort(key=lambda x: x.get(period_key, 0), reverse=True)

    return sector_stocks[:top_n]


def get_swing_focus_sectors(sector_performance: List[Dict], top_n: int = 3) -> List[Dict]:
    """
    Identify sectors to focus on for swing trading.

    Criteria:
    - Strong 3-month performance (trending sectors)
    - Positive weekly momentum (not exhausted)
    - Multiple stocks showing strength
    """
    focus_sectors = []

    for sector in sector_performance[:top_n * 2]:  # Check more to filter
        # Skip sectors with too few stocks
        if sector['stock_count'] < 3:
            continue

        # Must have positive 3m return
        if sector['avg_3m_return'] <= 0:
            continue

        # Weekly momentum should not be strongly negative
        if sector['avg_1w_return'] < -5:
            continue

        momentum = get_sector_momentum_score(sector)

        focus_sectors.append({
            **sector,
            'momentum': momentum,
            'recommendation': 'Focus' if 'Bullish' in momentum else 'Watch'
        })

        if len(focus_sectors) >= top_n:
            break

    return focus_sectors


if __name__ == '__main__':
    # Test the strategy
    print("Testing Swing Strategy on RELIANCE.NS...")
    result = backtest_strategy('RELIANCE.NS', period='2y')
    print(f"Total Return: {result.get('total_return_pct', 0):.2f}%")
    print(f"Win Rate: {result.get('win_rate', 0):.2f}%")
    print(f"Total Trades: {result.get('total_trades', 0)}")
