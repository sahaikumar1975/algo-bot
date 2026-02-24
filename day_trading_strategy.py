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

try:
    from custom_hmm import CustomGaussianHMM
except ImportError:
    CustomGaussianHMM = None

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

def calculate_ema(df: pd.DataFrame, period: int = 9) -> pd.DataFrame:
    """Calculate EMA dynamically."""
    df[f'EMA{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
    # For backwards compatibility where code expects EMA9 specifically if using default
    if period == 9 and 'EMA9' not in df.columns:
        pass # It is already created
    return df

# Alias for backward compatibility
calculate_9ema = lambda df: calculate_ema(df, period=9)

def fetch_data(ticker: str, period: str = '5d', broker=None, interval_fyers="5", interval_yf="5m") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Intraday and Daily data."""
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
            intraday_df = broker.get_history(fyers_sym, interval_fyers, start_intra, today_str, date_format='1')
            
            # --- CRITICAL FIX: Check Data Freshness ---
            if not intraday_df.empty:
                last_dt = intraday_df.index[-1].date()
                curr_dt = datetime.now().date()
                # If market is open (>= 09:15) and last data is NOT from today, it's stale
                is_market_open = datetime.now().time() >= datetime.strptime("09:15", "%H:%M").time()
                
                if is_market_open and last_dt < curr_dt:
                     logging.warning(f"⚠️ Fyers Data Stale for {ticker}: Last {last_dt}, Curr {curr_dt}. Forcing Fallback.")
                     raise Exception("Stale Data")
                
                # --- HYBRID FETCH: Append Real-Time Quote (LTP) if History is Lagging ---
                try:
                    quotes = broker.get_quotes([fyers_sym])
                    if quotes and len(quotes) > 0:
                        q = quotes[0]
                        # Fyers V3 Quote keys: 'lp'=LTP, 'v'=Vol, 't'=Timestamp (Epoch), 'o','h','l'
                        if 't' in q and 'lp' in q:
                            quote_ts = pd.to_datetime(q['t'], unit='s', utc=True).tz_convert('Asia/Kolkata')
                            last_hist_ts = intraday_df.index[-1]
                            
                            # If Quote is newer than last candle (by at least 1 min to avoid dupes)
                            if quote_ts > last_hist_ts:
                                logging.info(f"Hybrid: Appending Latest Quote for {ticker} @ {q['lp']}")
                                new_row = pd.DataFrame([{
                                    'Open': float(q.get('o', q['lp'])),
                                    'High': float(q.get('h', q['lp'])),
                                    'Low': float(q.get('l', q['lp'])),
                                    'Close': float(q['lp']),
                                    'Volume': float(q.get('v', 0))
                                }], index=[quote_ts])
                                intraday_df = pd.concat([intraday_df, new_row])
                except Exception as e:
                    logging.error(f"Hybrid Quote Error {ticker}: {e}")

            if not daily_df.empty and not intraday_df.empty:
                return daily_df, intraday_df
        except Exception as e:
            logging.warning(f"Fyers fetch failed for {ticker}: {e}")
            
    for attempt in range(3):
        try:
            daily_df = yf.download(ticker, period='1y', interval='1d', progress=False, timeout=10)
            intraday_df = yf.download(ticker, period='60d', interval=interval_yf, progress=False, timeout=10)
            if not daily_df.empty and not intraday_df.empty:
                break
        except Exception as e:
            time.sleep(2)
            
    if isinstance(daily_df.columns, pd.MultiIndex): daily_df.columns = daily_df.columns.get_level_values(0)
    if isinstance(intraday_df.columns, pd.MultiIndex): intraday_df.columns = intraday_df.columns.get_level_values(0)
        
    return daily_df, intraday_df

# --- INDICES STRATEGY (3-Candle Cluster) ---
def check_3_candle_setup(df_slice, sl_method='CLUSTER', limit_buffer=5.0, ema_col='EMA9'):
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

    if pd.isna(c1[ema_col]) or pd.isna(c2[ema_col]) or pd.isna(c3[ema_col]): return None

    # --- BUY SETUP (CE) ---
    # 1. Fresh Cross: C1 crosses UP (Open < EMA, Close > EMA)
    # 2. Previous Candle must be BELOW EMA (to ensure it's a fresh move)
    c1_crossover = c1['Open'] < c1[ema_col] and c1['Close'] > c1[ema_col]
    prev_below = c_prev['Close'] < c_prev[ema_col]
    # If c1 opened below, prev close likely below, but checking explicitly is safer
    
    all_above = c1['Close'] > c1[ema_col] and c2['Close'] > c2[ema_col] and c3['Close'] > c3[ema_col]
    
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
    c1_cross_sell = c1['Open'] > c1[ema_col] and c1['Close'] < c1[ema_col]
    prev_above = c_prev['Close'] > c_prev[ema_col]
    
    all_below = c1['Close'] < c1[ema_col] and c2['Close'] < c2[ema_col] and c3['Close'] < c3[ema_col]
    
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
    intraday_df = calculate_ema(intraday_df, period=9) # Legacy function call context
    
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

def train_hmm_model(daily_df: pd.DataFrame):
    """Train HMM model on daily returns and volatility to detect regime."""
    if CustomGaussianHMM is None or len(daily_df) < 50:
        return None
        
    df = daily_df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=5).std()
    df = df.dropna()
    
    if len(df) < 20:
        return None
        
    X = df[['Returns', 'Volatility']].values
    
    model = CustomGaussianHMM(n_components=3, n_iter=100, random_state=42)
    try:
        model.fit(X)
        return model
    except Exception as e:
        logging.error(f"HMM Training failed: {e}")
        return None

def get_hmm_regime(model, daily_df: pd.DataFrame) -> str:
    """Predict current regime ('BULLISH', 'BEARISH', 'CHOPPY')."""
    if model is None or len(daily_df) < 10:
        return 'UNKNOWN'
        
    df = daily_df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=5).std()
    df = df.dropna()
    
    if len(df) == 0:
        return 'UNKNOWN'
        
    X = df[['Returns', 'Volatility']].values
    try:
        hidden_states = model.predict(X)
        # Ensure hidden_states is easily indexable (numpy flatten)
        hidden_states = np.array(hidden_states).flatten()
        current_state = hidden_states[-1]
        
        state_returns = []
        df_returns = df['Returns'].values.flatten()
        for i in range(model.n_components):
            mask = (hidden_states == i)
            state_count = np.sum(mask)
            mean_ret = np.mean(df_returns[mask]) if state_count > 0 else 0
            state_returns.append(mean_ret)
            
        sorted_states = np.argsort(state_returns)
        
        if current_state == sorted_states[2]:
            return 'BULLISH'
        elif current_state == sorted_states[0]:
            return 'BEARISH'
        else:
            return 'CHOPPY'
    except Exception as e:
        logging.error(f"Error predicting HMM regime: {e}")
        return 'UNKNOWN'

# --- NEW: HMM + PRICE ACTION (ORB) STRATEGY ---
def check_orb_breakout_setup(df_slice, start_hour=9, start_min=15, end_hour=9, end_min=30):
    """
    Check for an Opening Range Breakout (ORB) on 15m candles.
    The "Opening Range" is established between start and end times.
    A breakout occurs when a candle CLOSES outside this range after the end time.
    """
    if df_slice.empty: return None
    
    # Ensure current candle is from today
    curr = df_slice.iloc[-1]
    curr_time = curr.name
    
    # Try to grab data just for the current date to establish ORB
    try:
        current_date_data = df_slice[df_slice.index.date == curr_time.date()]
        if current_date_data.empty: return None
    except:
        return None
        
    start_time_str = f"{start_hour:02d}:{start_min:02d}:00"
    end_time_str = f"{end_hour:02d}:{end_min:02d}:00"
    
    # Filter for the opening range candles
    opening_range_data = current_date_data.between_time(start_time_str, end_time_str)
    
    # Needs at least some candles to form a range (e.g. 9:15, 9:30, 9:45, 10:00 for a 15 min chart up to 10:15)
    if opening_range_data.empty or len(opening_range_data) < 2:
        return None
        
    # Calculate ORB High and Low
    orb_high = opening_range_data['High'].max()
    orb_low = opening_range_data['Low'].min()
    
    # We only look for breakouts AFTER the opening range is complete (e.g., after 10:15)
    if curr_time.time() <= pd.Timestamp(end_time_str).time():
        return None
        
    # We also want to know if a breakout ALREADY happened earlier in the day to avoid re-entering late.
    # For a pure system, we only take the FIRST candle that closes outside.
    # Check all candles today after the ORB end time up to the current one.
    after_orb_data = current_date_data.between_time(
        (pd.Timestamp(end_time_str) + pd.Timedelta(minutes=1)).time(), 
        curr_time.time()
    )
    
    if after_orb_data.empty: return None
    
    # Is the current candle the FIRST one to close outside the range today?
    # Or did a previous candle already breach it?
    previous_candles = after_orb_data.iloc[:-1]
    
    if not previous_candles.empty:
        prev_breakout_up = (previous_candles['Close'] > orb_high).any()
        prev_breakout_down = (previous_candles['Close'] < orb_low).any()
        if prev_breakout_up or prev_breakout_down:
            return None # A breakout already triggered earlier today. Wait for tomorrow.

    # Current Candle Breakout Check
    # SL is the midpoint of the ORB range to keep risk tight but give it breathing room.
    midpoint = (orb_high + orb_low) / 2.0
    
    # RISK CHECK: If the Opening Range is excessively massive (> 1.5% of the stock price), skip it.
    # This prevents taking breakouts on days with wild 9:15-10:15 volatility that creates an un-tradeable huge SL.
    orb_size_pct = (orb_high - orb_low) / orb_low * 100
    if orb_size_pct > 1.5:
        return None
    
    # LONG BREAKOUT
    if curr['Close'] > orb_high:
        risk = curr['Close'] - midpoint
        if risk <= 0: return None
        return {
            'Signal': 'LONG', 
            'Entry': curr['Close'], 
            'SL': midpoint, 
            'Time': curr_time,
            'OR_High': orb_high,
            'OR_Low': orb_low
        }
        
    # SHORT BREAKOUT
    elif curr['Close'] < orb_low:
        risk = midpoint - curr['Close']
        if risk <= 0: return None
        return {
            'Signal': 'SHORT', 
            'Entry': curr['Close'], 
            'SL': midpoint, 
            'Time': curr_time,
            'OR_High': orb_high,
            'OR_Low': orb_low
        }
        
    return None

def backtest_9ema_strategy(ticker):
    # Just a placeholder if called directly
    pass

if __name__ == "__main__":
    pass
