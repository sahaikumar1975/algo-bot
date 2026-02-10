"""
Day Trading Strategy Module
---------------------------
Contains strategies for:
1. Indices (9EMA 3-Candle Cluster)
2. Stocks (CPR + RSI + VWAP)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import time
import logging

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate VWAP (Intraday)."""
    v = df['Volume']
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    return df

    df['VWAP'] = (tp * v).cumsum() / v.cumsum()
    return df

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average Directional Index (ADX)."""
    df = df.copy()
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    df['+DM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['(-DM)'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                           np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    df['TR14'] = df['TR'].rolling(window=period).sum()
    df['+DM14'] = df['+DM'].rolling(window=period).sum()
    df['(-DM14)'] = df['(-DM)'].rolling(window=period).sum()

    df['+DI14'] = 100 * (df['+DM14'] / df['TR14'])
    df['(-DI14)'] = 100 * (df['(-DM14)'] / df['TR14'])
    
    df['DX'] = 100 * abs((df['+DI14'] - df['(-DI14)']) / (df['+DI14'] + df['(-DI14)']))
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    return df

def detect_market_regime(df: pd.DataFrame) -> str:
    """
    Detect Market Regime based on ADX.
    ADX > 25: Trending (SURFER)
    ADX <= 25: Choppy (SNIPER)
    """
    if len(df) < 30 or pd.isna(df.iloc[-1]['ADX']): return 'SNIPER' # Fallback
    
    adx = df.iloc[-1]['ADX']
    if adx > 25: return 'SURFER'
    return 'SNIPER'

def calculate_cpr(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate CPR (Central Pivot Range)."""
    df = daily_df.copy()
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['BC'] = (df['High'] + df['Low']) / 2
    df['TC'] = (df['Pivot'] - df['BC']) + df['Pivot']
    return df[['Pivot', 'BC', 'TC']].shift(1)

def calculate_9ema(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate 9 EMA."""
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    return df

def fetch_data(ticker: str, period: str = '5d', broker=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Intraday (5m) and Daily data."""
    daily_df = pd.DataFrame()
    intraday_df = pd.DataFrame()
    
    if broker:
        try:
            # Convert Yahoo Ticker to Fyers Symbol
            fyers_sym = ticker
            if ticker == '^NSEI': fyers_sym = "NSE:NIFTY50-INDEX"
            elif ticker == '^NSEBANK': fyers_sym = "NSE:NIFTYBANK-INDEX"
            elif ticker.endswith('.NS'):
                 fyers_sym = f"NSE:{ticker.replace('.NS', '')}-EQ"
            
            today = datetime.now()
            today_str = today.strftime("%Y-%m-%d")
            start_daily = (today - timedelta(days=365)).strftime("%Y-%m-%d")
            daily_df = broker.get_history(fyers_sym, "1D", start_daily, today_str, date_format='1')
            start_intra = (today - timedelta(days=59)).strftime("%Y-%m-%d")
            intraday_df = broker.get_history(fyers_sym, "5", start_intra, today_str, date_format='1')
            if not daily_df.empty and not intraday_df.empty:
                return daily_df, intraday_df
        except Exception as e:
            logging.warning(f"Fyers fetch failed for {ticker}: {e}")
            
    for attempt in range(3):
        try:
            daily_df = yf.download(ticker, period='1y', interval='1d', progress=False, timeout=10)
            intraday_df = yf.download(ticker, period='60d', interval='5m', progress=False, timeout=10)
            if not daily_df.empty and not intraday_df.empty:
                break
        except Exception as e:
            time.sleep(2)
            
    if isinstance(daily_df.columns, pd.MultiIndex): daily_df.columns = daily_df.columns.get_level_values(0)
    if isinstance(intraday_df.columns, pd.MultiIndex): intraday_df.columns = intraday_df.columns.get_level_values(0)
        
    return daily_df, intraday_df

# --- INDICES STRATEGY (3-Candle Cluster) ---
def check_3_candle_setup(df_slice, sl_method='CLUSTER', limit_buffer=5.0):
    """
    Check for 3-Candle Cluster Setup.
    sl_method: 'CLUSTER' (Lowest Low/Highest High of 3) or 'REF' (High/Low of Signal Candle).
    limit_buffer: Buffer points to add/subtract from SL.
    """
    if len(df_slice) < 4: return None
    
    # Candles: 0-2 (Cluster), 3 (Current)
    # Candles: 0-2 (Cluster), 3 (Current)
    # We need Previous Candle (c_prev) to ensure Fresh Cross
    if len(df_slice) < 5: return None
    
    c_prev = df_slice.iloc[-5]
    c1 = df_slice.iloc[-4]
    c2 = df_slice.iloc[-3]
    c3 = df_slice.iloc[-2]
    curr = df_slice.iloc[-1]
    
    # Intraday Check: Ensure all candles are from TODAY (User Request)
    try:
        # Convert timestamp to date for comparison
        d1 = c1.name.date()
        d2 = c2.name.date()
        d3 = c3.name.date()
        d4 = curr.name.date()
        
        if not (d1 == d2 == d3 == d4):
            return None
    except Exception as e:
        # logging.warning(f"Date check error: {e}")
        pass

    if pd.isna(c1['EMA9']) or pd.isna(c2['EMA9']) or pd.isna(c3['EMA9']): return None

    # --- BUY SETUP (CE) ---
    # 1. Fresh Cross: C1 crosses UP (Open < EMA, Close > EMA)
    # 2. Previous Candle must be BELOW EMA (to ensure it's a fresh move)
    c1_crossover = c1['Open'] < c1['EMA9'] and c1['Close'] > c1['EMA9']
    prev_below = c_prev['Close'] < c_prev['EMA9']
    # If c1 opened below, prev close likely below, but checking explicitly is safer
    
    all_above = c1['Close'] > c1['EMA9'] and c2['Close'] > c2['EMA9'] and c3['Close'] > c3['EMA9']
    
    if c1_crossover and prev_below and all_above:
        reds = [c for c in [c1, c2, c3] if c['Close'] < c['Open']]
        
        if len(reds) >= 1:
            ref_candle = reds[-1]
            entry_level = ref_candle['High']
            
            sl = 0
            if sl_method == 'REF':
                sl = ref_candle['Low']
            else:
                sl = min(c1['Low'], c2['Low'], c3['Low'])
            
            sl -= limit_buffer
            
            if curr['Close'] > entry_level:
                 return {'Signal': 'LONG', 'Entry': entry_level, 'SL': sl, 'Time': curr.name}
                 
    # --- SELL SETUP (PE) ---
    # 1. Fresh Cross: C1 crosses DOWN (Open > EMA, Close < EMA)
    # 2. Previous Candle must be ABOVE EMA
    c1_cross_sell = c1['Open'] > c1['EMA9'] and c1['Close'] < c1['EMA9']
    prev_above = c_prev['Close'] > c_prev['EMA9']
    
    all_below = c1['Close'] < c1['EMA9'] and c2['Close'] < c2['EMA9'] and c3['Close'] < c3['EMA9']
    
    if c1_cross_sell and prev_above and all_below:
        greens = [c for c in [c1, c2, c3] if c['Close'] > c['Open']]
        
        if len(greens) >= 1:
            ref_candle = greens[-1]
            entry_level = ref_candle['Low']
            
            sl = 0
            if sl_method == 'REF':
                sl = ref_candle['High']
            else:
                sl = max(c1['High'], c2['High'], c3['High'])
            
            sl += limit_buffer
                
            if curr['Close'] < entry_level:
                return {'Signal': 'SHORT', 'Entry': entry_level, 'SL': sl, 'Time': curr.name}
                
    return None

def backtest_day_strategy(ticker, initial_capital=100000, risk_per_trade=0.01, max_daily_loss=0.02, strategy_mode='SNIPER', exclude_days=None):
    """Backtest Stock Strategy (CPR)."""
    daily_df, intraday_df = fetch_data(ticker)
    if daily_df.empty or intraday_df.empty: return []

    # Calculate Indicators
    intraday_df = calculate_9ema(intraday_df)
    
    trades = []
    capital = initial_capital
    
    # Track trades per day
    daily_trade_counts = {} # {date_str: count}
    MAX_TRADES_PER_DAY = 2 # Hardcoded per requirement or config
    
    # State
    position = 0 # 0=Flat, 1=Long, -1=Short
    entry_price = 0
    sl = 0
    target = 0
    entry_time = None
    
    # Iterate candle by candle
    # Need at least 4 candles for setup
    for i in range(4, len(intraday_df)):
        curr = intraday_df.iloc[i]
        curr_time = curr.name
        curr_date_str = curr_time.strftime('%Y-%m-%d')
        
        # 1. Manage Open Position
        if position != 0:
            exit_reason = None
            exit_price = 0
            
            # Check OHLC of CURRENT candle for SL/Target fill
            # Optimistic/Pessimistic assumption: 
            # If SL hit in candle, we exit at SL. If Target hit, exit at Target.
            # If both hit? worst case SL.
            
            low = curr['Low']
            high = curr['High']
            close = curr['Close']
            
            if position == 1: # Long
                if low <= sl: 
                    exit_reason = "SL Hit"
                    exit_price = sl
                elif strategy_mode == 'SNIPER' and high >= target:
                    exit_reason = "Target Hit"
                    exit_price = target
                # HYBRID EXIT for Sniper: Also check 9EMA Trail
                elif strategy_mode == 'SNIPER' and close < curr['EMA9']:
                    exit_reason = "Trailing Exit (Close < 9EMA)"
                    exit_price = close
                elif strategy_mode == 'SURFER' and close < curr['EMA9']:
                    exit_reason = "Trailing Exit (Close < 9EMA)"
                    exit_price = close
                elif curr_time.hour >= 15 and curr_time.minute >= 15: # EOD
                     exit_reason = "EOD Square Off"
                     exit_price = close
                     
            elif position == -1: # Short
                if high >= sl:
                    exit_reason = "SL Hit"
                    exit_price = sl
                elif strategy_mode == 'SNIPER' and low <= target:
                    exit_reason = "Target Hit"
                    exit_price = target
                # HYBRID EXIT for Sniper: Also check 9EMA Trail
                elif strategy_mode == 'SNIPER' and close > curr['EMA9']:
                    exit_reason = "Trailing Exit (Close > 9EMA)"
                    exit_price = close
                elif strategy_mode == 'SURFER' and close > curr['EMA9']:
                    exit_reason = "Trailing Exit (Close > 9EMA)"
                    exit_price = close
                elif curr_time.hour >= 15 and curr_time.minute >= 15:
                     exit_reason = "EOD Square Off"
                     exit_price = close
            
            if exit_reason:
                pnl = (exit_price - entry_price) * (1 if position == 1 else -1)
                trades.append({
                    'Entry Time': entry_time,
                    'Exit Time': curr_time,
                    'Signal': 'LONG' if position == 1 else 'SHORT',
                    'Entry': entry_price,
                    'Exit': exit_price,
                    'PnL': pnl,
                    'Reason': exit_reason,
                    'Mode': strategy_mode
                })
                position = 0
                entry_price = 0
                sl = 0
                target = 0
                entry_time = None
                continue # Trade closed, move to next candle
                
        # 2. Check for New Entry (If Flat)
        # Check Limits
        day_trades = daily_trade_counts.get(curr_date_str, 0)
        if day_trades >= MAX_TRADES_PER_DAY: continue
        
        # Check Time (No new trades after 3:00 PM)
        if curr_time.hour >= 15: continue
        
        # Check Signal using LAST 4 candles (Ending at i)
        # i is current candle (forming or closed?). In backtest loop, 'curr' is the candle at index i.
        # check_3_candle_setup expects a slice where the last candle is the 'current' one.
        # We use [i-3 : i+1] to get 4 candles: i-3, i-2, i-1, i
        
        df_slice = intraday_df.iloc[i-3 : i+1]
        
        # Use SL Method 'REF' as per Live Bot Stock Logic
        signal_data = check_3_candle_setup(df_slice, sl_method='REF', limit_buffer=0.0)
        
        if signal_data:
            signal_type = signal_data['Signal']
            limit_entry = signal_data['Entry'] # We enter if price crosses this
            stop_loss = signal_data['SL']
            
            # Entry Logic:
            # We are at candle 'i'. The signal is based on i (Current) breaking High/Low of prev candles?
            # Wait, check_3_candle_setup logic: 
            # "if curr['Close'] > entry_level" -> It checks if the CURRENT candle closed above entry.
            # If so, we assume we entered AT THE CLOSE of this candle? 
            # Or we assume we entered the moment it crossed? 
            # For simplicity in 5m backtest: Enter at CLOSE of Signal Candle if condition met.
            
            # If signal returned, it means condition MET.
            entry_p = curr['Close'] # Market order at close check
            # Or use the specific Entry Level? 
            # Let's use Close to be safe/realistic for "Close > Level" confirmation.
            
            risk = abs(entry_p - stop_loss)
            if risk == 0: continue
            
            target_p = entry_p + (risk * 2 * (1 if signal_type == 'LONG' else -1))
            
            position = 1 if signal_type == 'LONG' else -1
            entry_price = entry_p
            sl = stop_loss
            target = target_p
            entry_time = curr_time
            
            # Increment daily count
            daily_trade_counts[curr_date_str] = day_trades + 1
            
    return trades

    return None

# --- STOCK STRATEGY (CPR + RSI + VWAP) ---
def check_stock_signal(curr, prev, cpr_data):
    """
    Check for Stock Strategy Signals.
    curr, prev: Intraday Rows (with RSI, VWAP).
    cpr_data: {'Pivot': val, 'TC': val, 'BC': val} for the day.
    """
    if pd.isna(curr['RSI']) or pd.isna(curr['VWAP']): return None
    
    rsi = curr['RSI']
    close = curr['Close']
    vwap = curr['VWAP']
    pivot = cpr_data['Pivot']
    tc = cpr_data['TC']
    bc = cpr_data['BC']
    
    # LONG: Close > VWAP, RSI > 55, Close > TC (Bullish CPR Break or Support)
    # Simplified Trend Following
    if close > vwap and rsi > 55 and close > tc:
        if prev['Close'] <= prev['VWAP'] or prev['RSI'] <= 55: # Crossover
             sl = min(prev['Low'], bc)
             return {'Signal': 'LONG', 'Entry': close, 'SL': sl}
             
    # SHORT: Close < VWAP, RSI < 45, Close < BC
    if close < vwap and rsi < 45 and close < bc:
        if prev['Close'] >= prev['VWAP'] or prev['RSI'] >= 45:
             sl = max(prev['High'], tc)
             return {'Signal': 'SHORT', 'Entry': close, 'SL': sl}
             
    return None

def check_9ema_signal(curr, prev):
    # LEGACY / Fallback
    if pd.isna(prev['EMA9']): return None
    ema9 = prev['EMA9']
    if prev['Low'] > ema9 and curr['Low'] < prev['Low']:
        return {'Signal': 'SHORT', 'Entry': prev['Low'], 'SL': prev['High']}
    elif prev['High'] < ema9 and curr['High'] > prev['High']:
        return {'Signal': 'LONG', 'Entry': prev['High'], 'SL': prev['Low']}
    return None

def backtest_9ema_strategy(ticker):
    # Just a placeholder if called directly
    pass

if __name__ == "__main__":
    pass
