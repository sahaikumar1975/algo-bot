import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional


def fetch_ohlc(ticker: str, period: str = "180d", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLC data for a ticker using yfinance.

    Returns a DataFrame indexed by date with columns: Open, High, Low, Close, Volume
    """
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval)
    if df.empty:
        return df
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    return df


def add_sma(df: pd.DataFrame, window: int, column: str = 'Close', name: Optional[str] = None) -> pd.DataFrame:
    if name is None:
        name = f'SMA{window}'
    df[name] = df[column].rolling(window=window, min_periods=1).mean()
    return df


def detect_sma21_50_crossover(df: pd.DataFrame) -> Dict:
    """Detect a completed SMA21 crossing above SMA50 on the last closed candle.

    Conditions:
    - previous bar: SMA21 <= SMA50
    - last bar: SMA21 > SMA50 (crossover happened on last closed candle)
    - both SMA21 and SMA50 > SMA200 on last bar
    - last close > SMA21 and > SMA50 (crossover candle completed bullish)

    Returns a dict with signal boolean and metadata.
    """
    out = {'signal': False}
    if df is None or df.shape[0] < 3:
        return out

    df = df.copy()
    df = add_sma(df, 21)
    df = add_sma(df, 50)
    df = add_sma(df, 200)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    prev_s21 = prev['SMA21']
    prev_s50 = prev['SMA50']
    last_s21 = last['SMA21']
    last_s50 = last['SMA50']
    last_s200 = last['SMA200']

    # crossover
    crossed = (prev_s21 <= prev_s50) and (last_s21 > last_s50)
    above_200 = (last_s21 > last_s200) and (last_s50 > last_s200)
    closed_above = (last['Close'] > last_s21) and (last['Close'] > last_s50)

    if crossed and above_200 and closed_above:
        out['signal'] = True
        out['date'] = last.name
        out['close'] = float(last['Close'])
        out['sma21'] = float(last_s21)
        out['sma50'] = float(last_s50)
        out['sma200'] = float(last_s200)
    return out


def detect_close_above_sma21(df: pd.DataFrame) -> Dict:
    """Detect stocks just closing above SMA21 with price above SMA50 and SMA200.

    Conditions:
    - Previous day close was BELOW SMA21
    - Current day close is ABOVE SMA21 (just crossed above)
    - Current close > SMA50 (above medium-term trend)
    - Current close > SMA200 (above long-term trend)
    - SMA200 is ascending (current SMA200 > previous SMA200)

    Returns a dict with signal boolean and metadata.
    """
    out = {'signal': False, 'signal_type': 'close_above_sma21'}
    if df is None or df.shape[0] < 3:
        return out

    df = df.copy()
    df = add_sma(df, 21)
    df = add_sma(df, 50)
    df = add_sma(df, 200)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    last_close = last['Close']
    prev_close = prev['Close']
    last_s21 = last['SMA21']
    prev_s21 = prev['SMA21']
    last_s50 = last['SMA50']
    last_s200 = last['SMA200']
    prev_s200 = prev['SMA200']

    # Check conditions:
    # 1. Previous close was below SMA21
    # 2. Current close is above SMA21 (just crossed)
    # 3. Current close is above SMA50
    # 4. Current close is above SMA200
    # 5. SMA200 is ascending (uptrend)
    prev_below_sma21 = prev_close <= prev_s21
    curr_above_sma21 = last_close > last_s21
    curr_above_sma50 = last_close > last_s50
    curr_above_sma200 = last_close > last_s200
    sma200_ascending = last_s200 > prev_s200

    if prev_below_sma21 and curr_above_sma21 and curr_above_sma50 and curr_above_sma200 and sma200_ascending:
        out['signal'] = True
        out['date'] = last.name
        out['close'] = float(last_close)
        out['sma21'] = float(last_s21)
        out['sma50'] = float(last_s50)
        out['sma200'] = float(last_s200)
        # Calculate distance from SMA21 as percentage
        out['dist_sma21_pct'] = float((last_close - last_s21) / last_s21 * 100)
    return out


def detect_volume_breakout(df: pd.DataFrame, vol_multiplier: float = 1.5, vol_avg_period: int = 20) -> Dict:
    """Detect volume breakout with SMA alignment (SMA21 > SMA50 > SMA200).

    Conditions:
    - Previous day volume > vol_multiplier * average volume (default 1.5x)
    - SMA21 > SMA50 > SMA200 (bullish alignment)
    - Close > SMA21 (price above short-term trend)

    Args:
        df: DataFrame with OHLC and Volume data
        vol_multiplier: Volume must be this many times the average (default 1.5)
        vol_avg_period: Period for calculating average volume (default 20 days)

    Returns a dict with signal boolean and metadata.
    """
    out = {'signal': False, 'signal_type': 'volume_breakout'}
    if df is None or df.shape[0] < vol_avg_period + 1:
        return out

    df = df.copy()
    df = add_sma(df, 21)
    df = add_sma(df, 50)
    df = add_sma(df, 200)

    # Calculate average volume (excluding the last bar)
    df['VolAvg'] = df['Volume'].rolling(window=vol_avg_period, min_periods=vol_avg_period).mean()

    last = df.iloc[-1]
    last_s21 = last['SMA21']
    last_s50 = last['SMA50']
    last_s200 = last['SMA200']
    last_vol = last['Volume']
    avg_vol = df['VolAvg'].iloc[-2]  # Use previous day's average to compare

    if pd.isna(avg_vol) or avg_vol == 0:
        return out

    # Check conditions
    sma_aligned = (last_s21 > last_s50) and (last_s50 > last_s200)
    volume_breakout = last_vol > (vol_multiplier * avg_vol)
    close_above_sma21 = last['Close'] > last_s21

    if sma_aligned and volume_breakout and close_above_sma21:
        out['signal'] = True
        out['date'] = last.name
        out['close'] = float(last['Close'])
        out['sma21'] = float(last_s21)
        out['sma50'] = float(last_s50)
        out['sma200'] = float(last_s200)
        out['volume'] = float(last_vol)
        out['avg_volume'] = float(avg_vol)
        out['vol_ratio'] = float(last_vol / avg_vol)
    return out


def screen_tickers(tickers, period: str = '180d', interval: str = '1d', include_df: bool = False):
    """Screen a list of tickers for SMA crossover signals.

    Args:
        tickers: List of ticker symbols to screen
        period: Historical period (e.g., '180d')
        interval: Data interval (e.g., '1d')
        include_df: If True, include the OHLC DataFrame with SMAs for charting

    Returns:
        List of dicts with signal metadata. If include_df=True, each dict
        also contains 'ohlc_df' with the full DataFrame including SMA columns.
    """
    results = []
    for t in tickers:
        try:
            df = fetch_ohlc(t, period=period, interval=interval)
            if df.empty:
                continue
            sig = detect_sma21_50_crossover(df)
            if sig.get('signal'):
                result = {
                    'ticker': t,
                    'signal_type': 'crossover',
                    'date': sig.get('date'),
                    'close': sig.get('close'),
                    'sma21': sig.get('sma21'),
                    'sma50': sig.get('sma50'),
                    'sma200': sig.get('sma200')
                }
                if include_df:
                    # Add SMAs to df for charting
                    df = add_sma(df, 21)
                    df = add_sma(df, 50)
                    df = add_sma(df, 200)
                    result['ohlc_df'] = df
                results.append(result)
        except Exception:
            # keep screening robust; skip tickers that fail
            continue
    return results


def screen_volume_breakout(tickers, period: str = '180d', interval: str = '1d',
                           include_df: bool = False, vol_multiplier: float = 1.5):
    """Screen a list of tickers for volume breakout signals with SMA alignment.

    Args:
        tickers: List of ticker symbols to screen
        period: Historical period (e.g., '180d')
        interval: Data interval (e.g., '1d')
        include_df: If True, include the OHLC DataFrame with SMAs for charting
        vol_multiplier: Volume threshold multiplier (default 1.5x average)

    Returns:
        List of dicts with signal metadata. If include_df=True, each dict
        also contains 'ohlc_df' with the full DataFrame including SMA columns.
    """
    results = []
    for t in tickers:
        try:
            df = fetch_ohlc(t, period=period, interval=interval)
            if df.empty:
                continue
            sig = detect_volume_breakout(df, vol_multiplier=vol_multiplier)
            if sig.get('signal'):
                result = {
                    'ticker': t,
                    'signal_type': 'volume_breakout',
                    'date': sig.get('date'),
                    'close': sig.get('close'),
                    'sma21': sig.get('sma21'),
                    'sma50': sig.get('sma50'),
                    'sma200': sig.get('sma200'),
                    'volume': sig.get('volume'),
                    'avg_volume': sig.get('avg_volume'),
                    'vol_ratio': sig.get('vol_ratio')
                }
                if include_df:
                    # Add SMAs to df for charting
                    df = add_sma(df, 21)
                    df = add_sma(df, 50)
                    df = add_sma(df, 200)
                    result['ohlc_df'] = df
                results.append(result)
        except Exception:
            # keep screening robust; skip tickers that fail
            continue
    return results


def screen_all(tickers, period: str = '180d', interval: str = '1d',
               include_df: bool = False, vol_multiplier: float = 1.5):
    """Screen tickers for close above SMA21 with price above SMA50 and SMA200.

    Args:
        tickers: List of ticker symbols to screen
        period: Historical period (e.g., '180d')
        interval: Data interval (e.g., '1d')
        include_df: If True, include the OHLC DataFrame with SMAs for charting
        vol_multiplier: Not used currently, kept for compatibility

    Returns:
        Tuple of (results, signal_type) where:
        - results: List of all matching stocks
        - signal_type: 'close_above_sma21' or None
    """
    results = []

    # Scan all tickers for close above SMA21 signal
    for t in tickers:
        try:
            df = fetch_ohlc(t, period=period, interval=interval)
            if df.empty:
                continue

            # Check for close above SMA21 signal
            sig = detect_close_above_sma21(df)
            if sig.get('signal'):
                result = {
                    'ticker': t,
                    'signal_type': 'close_above_sma21',
                    'date': sig.get('date'),
                    'close': sig.get('close'),
                    'sma21': sig.get('sma21'),
                    'sma50': sig.get('sma50'),
                    'sma200': sig.get('sma200'),
                    'dist_sma21_pct': sig.get('dist_sma21_pct')
                }
                if include_df:
                    df = add_sma(df, 21)
                    df = add_sma(df, 50)
                    df = add_sma(df, 200)
                    result['ohlc_df'] = df
                results.append(result)

        except Exception:
            continue

    signal_type = 'close_above_sma21' if results else None
    return results, signal_type


if __name__ == '__main__':
    # simple local test
    tickers = ['RELIANCE.NS', 'TCS.NS']
    matches = screen_tickers(tickers)
    print(matches)
