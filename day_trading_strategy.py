"""
Day Trading Strategy Module: CPR + RSI + Volume
Timeframe: 5 Minutes
Target: >50% Win Rate, 1:2 Risk:Reward
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index (RSI)."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_cpr(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate CPR (Central Pivot Range) from Daily Data.
    Returns DataFrame with 'Pivot', 'BC', 'TC' for the NEXT trading day.
    """
    df = daily_df.copy()
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['BC'] = (df['High'] + df['Low']) / 2
    df['TC'] = (df['Pivot'] - df['BC']) + df['Pivot']
    return df[['Pivot', 'BC', 'TC']].shift(1)

def fetch_data(ticker: str, period: str = '5d') -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch Intraday (5m) and Daily data.
    """
    daily_df = yf.download(ticker, period='1y', interval='1d', progress=False)
    intraday_df = yf.download(ticker, period='60d', interval='5m', progress=False)
    
    if isinstance(daily_df.columns, pd.MultiIndex):
        daily_df.columns = daily_df.columns.get_level_values(0)
    if isinstance(intraday_df.columns, pd.MultiIndex):
        intraday_df.columns = intraday_df.columns.get_level_values(0)
        
    return daily_df, intraday_df

def get_atm_strike(price, ticker):
    """Calculate ATM Strike Price."""
    if 'BANK' in ticker.upper():
        return int(round(price / 100) * 100)
    return int(round(price / 50) * 50)

def backtest_day_strategy(ticker, initial_capital=100000, risk_per_trade=0.01, max_daily_loss=0.02, strategy_mode='SNIPER', exclude_days=None):
    """
    Backtest the CPR Breakout/Bounce Strategy on 5m data.
    strategy_mode: 'SNIPER' (Fixed Targets) or 'SURFER' (EMA 20 Trail)
    """
    """
    Backtest the CPR Breakout/Bounce Strategy on 5m data.
    """
    print(f"--- Starting Backtest for {ticker} ---")
    
    daily_df, intraday_df = fetch_data(ticker)
    
    if intraday_df.empty or daily_df.empty:
        print("Insufficient data.")
        return
    
    # 1. Prepare Daily CPR values
    daily_cpr = calculate_cpr(daily_df)
    
    # 2. Prepare Intraday Indicators
    intraday_df = add_rsi(intraday_df)
    intraday_df['Vol_SMA20'] = intraday_df['Volume'].rolling(20).mean()
    intraday_df['EMA20'] = intraday_df['Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate VWAP
    intraday_df['Cum_Vol'] = intraday_df.groupby(intraday_df.index.date)['Volume'].cumsum()
    intraday_df['Cum_Vol_Price'] = intraday_df.groupby(intraday_df.index.date).apply(lambda x: (x['Close'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
    intraday_df['VWAP'] = intraday_df['Cum_Vol_Price'] / intraday_df['Cum_Vol']
    
    # FIX: For Indices (Volume=0), VWAP is NaN. Use EMA20 as proxy.
    if ticker.startswith('^'):
        intraday_df['VWAP'] = intraday_df['EMA20']
    
    # Fill remaining NaNs (start of session)
    intraday_df['VWAP'] = intraday_df['VWAP'].fillna(intraday_df['EMA20'])
    
    # ATR for SL/Target
    high_low = intraday_df['High'] - intraday_df['Low']
    high_close = np.abs(intraday_df['High'] - intraday_df['Close'].shift())
    low_close = np.abs(intraday_df['Low'] - intraday_df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    intraday_df['ATR'] = true_range.rolling(14).mean()
    
    intraday_df['ATR'] = true_range.rolling(14).mean()
    
    # ADX Calculation (Trend Strength)
    plus_dm = intraday_df['High'] - intraday_df['High'].shift(1)
    minus_dm = intraday_df['Low'].shift(1) - intraday_df['Low']
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr_s = true_range.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr_s)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr_s)
    dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
    intraday_df['ADX'] = dx.rolling(14).mean()

    # Merge CPR into Intraday Data
    intraday_df['Date_Only'] = intraday_df.index.date
    daily_cpr['Date_Only'] = daily_cpr.index.date
    
    # Daily trend filter: Compare Today's Pivot vs Yesterday's Pivot
    daily_cpr['Prev_Pivot'] = daily_cpr['Pivot'].shift(1)
    
    pivot_map = daily_cpr.set_index('Date_Only')['Pivot'].to_dict()
    prev_pivot_map = daily_cpr.set_index('Date_Only')['Prev_Pivot'].to_dict()
    bc_map = daily_cpr.set_index('Date_Only')['BC'].to_dict()
    tc_map = daily_cpr.set_index('Date_Only')['TC'].to_dict()
    
    intraday_df['Pivot'] = intraday_df['Date_Only'].map(pivot_map)
    intraday_df['Prev_Pivot'] = intraday_df['Date_Only'].map(prev_pivot_map)
    intraday_df['BC'] = intraday_df['Date_Only'].map(bc_map)
    intraday_df['TC'] = intraday_df['Date_Only'].map(tc_map)
    
    # Filter valid data
    intraday_df.dropna(subset=['Pivot', 'Prev_Pivot', 'RSI', 'ATR', 'VWAP'], inplace=True)
    
    # Backtest Logic
    capital = initial_capital
    risk_per_trade_pct = risk_per_trade 
    max_daily_loss_pct = max_daily_loss
    
    position = 0 # 0, 1 (Long), -1 (Short)
    entry_price = 0.0
    stop_loss = 0.0
    target = 0.0
    quantity = 0
    
    trades = []
    wins = 0
    losses = 0
    
    daily_pnl = {} # Track PnL per day
    
    for i in range(1, len(intraday_df)):
        curr = intraday_df.iloc[i]
        prev = intraday_df.iloc[i-1]
        timestamp = intraday_df.index[i]
        
        # --- TRAILING STOP LOSS (TSL) LOGIC ---
        if position != 0:
            # SURFER MODE EXIT (EMA 20 Cross)
            if strategy_mode == 'SURFER':
                # Force Exit if Trend Bends
                trend_bent = False
                if position == 1 and curr['Close'] < curr['EMA20']: trend_bent = True
                elif position == -1 and curr['Close'] > curr['EMA20']: trend_bent = True
                
                if trend_bent:
                    pnl = (curr['Close'] - entry_price) * quantity * position
                     # OPTION SIMULATION
                    option_note = ""
                    if ticker.startswith('^'):
                        pnl = pnl * 0.5
                        strike = get_atm_strike(entry_price, ticker)
                        otype = "CE" if position == 1 else "PE"
                        option_note = f" [{strike} {otype}]"

                    capital += pnl
                    # Update daily PnL (Logic duplicated for brevity, ideally refactor)
                    daily_pnl[current_date] = daily_pnl.get(current_date, 0) + pnl
                    
                    trades.append({
                        'Time': timestamp, 'Symbol': ticker + option_note, 'Side': 'Long' if position == 1 else 'Short',
                        'Entry': entry_price, 'Exit': curr['Close'], 'Qty': quantity,
                        'PnL': pnl, 'Type': 'EMA20 Exit (Surfer)', 'Result': 'Win' if pnl > 0 else 'Loss'
                    })
                    position = 0
                    continue

            # SNIPER MODE TSL (Original Logic)
            elif strategy_mode == 'SNIPER':
                risk_unit = abs(entry_price - initial_stop_loss)
                
                if position == 1: # Long
                    if not sl_trailed:
                        # Trigger 1: Move to Breakeven at 1.25R
                        if curr['High'] >= (entry_price + (risk_unit * 1.25)):
                            stop_loss = entry_price
                            sl_trailed = True
                    
                    if sl_trailed and not sl_profit:
                        # Trigger 2: Move to 0.5R Profit at 1.5R
                        if curr['High'] >= (entry_price + (risk_unit * 1.5)):
                            stop_loss = entry_price + (risk_unit * 0.5)
                            sl_profit = True
                
                elif position == -1: # Short
                    if not sl_trailed:
                        # Trigger 1: Move to Breakeven at 1.25R
                        if curr['Low'] <= (entry_price - (risk_unit * 1.25)):
                            stop_loss = entry_price
                            sl_trailed = True
                    
                    if sl_trailed and not sl_profit:
                        # Trigger 2: Move to 0.5R Profit at 1.5R
                        if curr['Low'] <= (entry_price - (risk_unit * 1.5)):
                            stop_loss = entry_price - (risk_unit * 0.5)
                            sl_profit = True

        # Convert to IST for correct time filtering
        if timestamp.tzinfo is not None:
            timestamp = timestamp.tz_convert('Asia/Kolkata')
        
        current_date = timestamp.date()
        
        # Check Daily Loss Limit
        current_day_pnl = daily_pnl.get(current_date, 0)
        daily_loss_limit = -(capital * max_daily_loss_pct)
        
        if current_day_pnl <= daily_loss_limit:
            # FORCE EXIT if holding
            if position != 0:
                pnl = (curr['Close'] - entry_price) * quantity * position
                
                # OPTION SIMULATION (Delta 0.5)
                option_note = ""
                if ticker.startswith('^'):
                    pnl = pnl * 0.5 # Simulate ATM Option Delta
                    strike = get_atm_strike(entry_price, ticker)
                    otype = "CE" if position == 1 else "PE"
                    option_note = f" [{strike} {otype}]"

                capital += pnl
                current_day_pnl += pnl
                daily_pnl[current_date] = current_day_pnl
                trades.append({
                    'Time': timestamp, 'Symbol': ticker + option_note, 'Side': 'Long' if position == 1 else 'Short',
                    'Entry': entry_price, 'Exit': curr['Close'], 'Qty': quantity,
                    'PnL': pnl, 'Type': 'Daily Risk Stop', 'Result': 'Loss'
                })
                position = 0
            continue # Skip trading for rest of day
            
        # TIME FILTER: Only trade between 09:30 and 15:00
        current_time = timestamp.time()
        start_time = datetime.strptime("09:30", "%H:%M").time()
        last_trade_time = datetime.strptime("15:00", "%H:%M").time()
        exit_time = datetime.strptime("15:15", "%H:%M").time()
        
        if current_time >= exit_time:
            if position != 0:
                pnl = (curr['Close'] - entry_price) * quantity * position
                
                # OPTION SIMULATION (Delta 0.5)
                option_note = ""
                if ticker.startswith('^'):
                    pnl = pnl * 0.5
                    strike = get_atm_strike(entry_price, ticker)
                    otype = "CE" if position == 1 else "PE"
                    option_note = f" [{strike} {otype}]"

                capital += pnl
                # Update daily PnL
                daily_pnl[current_date] = daily_pnl.get(current_date, 0) + pnl
                
                res = 'Win' if pnl > 0 else 'Loss'
                if pnl > 0: wins += 1
                else: losses += 1
                trades.append({
                    'Time': timestamp, 'Symbol': ticker + option_note, 'Side': 'Long' if position == 1 else 'Short',
                    'Entry': entry_price, 'Exit': curr['Close'], 'Qty': quantity,
                    'PnL': pnl, 'Type': 'EOD Exit', 'Result': res
                })
                position = 0
            continue
            
        if current_time < start_time:
            continue
            
        upper_cpr = max(curr['TC'], curr['BC'])
        lower_cpr = min(curr['TC'], curr['BC'])
        
        # CPR Trend
        cpr_bullish = curr['Pivot'] > curr['Prev_Pivot']
        cpr_bearish = curr['Pivot'] < curr['Prev_Pivot']
        
        # Market structure check
        above_vwap = curr['Close'] > curr['VWAP']
        below_vwap = curr['Close'] < curr['VWAP']
        
        # NO EXISTING POSITION
        if position == 0:
            if current_time > last_trade_time: continue
            
            # WEEKDAY EXCLUSION FILTER
            if exclude_days and timestamp.strftime('%A') in exclude_days:
                continue

            # --- LONG SETUP ---
            cpr_breakout_long = (prev['Close'] < upper_cpr and curr['Close'] > upper_cpr)
            rsi_cross_long = (prev['RSI'] <= 60 and curr['RSI'] > 60)
            
            valid_long = (
                cpr_bullish and 
                above_vwap and 
                (cpr_breakout_long or rsi_cross_long) and
                (ticker.startswith('^') or curr['Volume'] > curr['Vol_SMA20']) and
                (abs(curr['TC'] - curr['BC']) < (curr['Close'] * 0.0025)) # Narrow CPR Filter (<0.25%)
            )
            
            if valid_long:
                entry_price = curr['Close']
                
                # FIXED POINT TARGETS FOR INDICES
                if ticker.startswith('^'):
                    if 'BANK' in ticker:
                        # Opt SL 35 / TP 70 -> Index SL 70 / TP 140
                        sl_points = 70
                        tp_points = 99999 if strategy_mode == 'SURFER' else 140 # Unlimited Target for Surfer
                    else:
                        # Opt SL 15 / TP 30 -> Index SL 30 / TP 60
                        sl_points = 30
                        tp_points = 99999 if strategy_mode == 'SURFER' else 60
                    stop_loss = entry_price - sl_points
                    target = entry_price + tp_points
                else: 
                    atr_sl = curr['ATR'] * 1.5
                    stop_loss = entry_price - atr_sl
                    target = entry_price + (atr_sl * 2) 
                
                # Position Sizing
                risk_amt = capital * risk_per_trade_pct
                risk_per_share = entry_price - stop_loss
                if risk_per_share > 0:
                    quantity = int(risk_amt / risk_per_share)
                    
                    # LOT SIZE ENFORCEMENT
                    if ticker.startswith('^'):
                        lot_size = 30 if 'BANK' in ticker else 65
                        if quantity < lot_size:
                            quantity = lot_size # Min 1 Lot
                        else:
                            quantity = (quantity // lot_size) * lot_size # Round to nearest lot
                            
                else:
                    quantity = 0
                
                if quantity > 0:
                    position = 1
                    initial_stop_loss = stop_loss
                    sl_trailed = False
                    sl_profit = False
                continue

            # --- SHORT SETUP ---
            cpr_breakdown_short = (prev['Close'] > lower_cpr and curr['Close'] < lower_cpr)
            rsi_cross_short = (prev['RSI'] >= 40 and curr['RSI'] < 40)
            
            valid_short = (
                cpr_bearish and 
                below_vwap and
                (cpr_breakdown_short or rsi_cross_short) and
                (ticker.startswith('^') or curr['Volume'] > curr['Vol_SMA20']) and
                (abs(curr['TC'] - curr['BC']) < (curr['Close'] * 0.0025)) # Narrow CPR Filter (<0.25%)
            )
            
            if valid_short:
                entry_price = curr['Close']
                
                # FIXED POINT TARGETS FOR INDICES
                if ticker.startswith('^'):
                    if 'BANK' in ticker:
                        sl_points = 70
                        tp_points = 99999 if strategy_mode == 'SURFER' else 140
                    else:
                        sl_points = 30
                        tp_points = 99999 if strategy_mode == 'SURFER' else 60
                    stop_loss = entry_price + sl_points
                    target = entry_price - tp_points
                else:
                    atr_sl = curr['ATR'] * 1.5
                    stop_loss = entry_price + atr_sl # SL is above entry for short
                    target = entry_price - (atr_sl * 2) 
                
                # Position Sizing
                risk_amt = capital * risk_per_trade_pct
                risk_per_share = stop_loss - entry_price
                if risk_per_share > 0:
                    quantity = int(risk_amt / risk_per_share)
                    
                    # LOT SIZE ENFORCEMENT
                    if ticker.startswith('^'):
                        lot_size = 30 if 'BANK' in ticker else 65
                        if quantity < lot_size:
                            quantity = lot_size # Min 1 Lot
                        else:
                            quantity = (quantity // lot_size) * lot_size # Round to nearest lot
                            
                else:
                    quantity = 0
                
                if quantity > 0:
                    position = -1
                    initial_stop_loss = stop_loss
                    sl_trailed = False
                    sl_profit = False
                continue
                
        # MANAGE EXISTING POSITION
        elif position == 1: # Long
            if curr['High'] >= target:
                pnl = (target - entry_price) * quantity
                
                # OPTION SIMULATION
                option_note = ""
                if ticker.startswith('^'):
                    pnl = pnl * 0.5
                    strike = get_atm_strike(entry_price, ticker)
                    option_note = f" [{strike} CE]"

                capital += pnl
                daily_pnl[current_date] = daily_pnl.get(current_date, 0) + pnl
                wins += 1
                trades.append({
                    'Time': timestamp, 'Symbol': ticker + option_note, 'Side': 'Long',
                    'Entry': entry_price, 'Exit': target, 
                    'Qty': quantity, 'PnL': pnl, 'Result': 'Win', 'Type': 'Target Hit'
                })
                position = 0
            elif curr['Low'] <= stop_loss:
                pnl = (stop_loss - entry_price) * quantity
                
                # OPTION SIMULATION
                option_note = ""
                if ticker.startswith('^'):
                    pnl = pnl * 0.5
                    strike = get_atm_strike(entry_price, ticker)
                    option_note = f" [{strike} CE]"

                capital += pnl
                daily_pnl[current_date] = daily_pnl.get(current_date, 0) + pnl
                losses += 1
                trades.append({
                    'Time': timestamp, 'Symbol': ticker + option_note, 'Side': 'Long',
                    'Entry': entry_price, 'Exit': stop_loss, 
                    'Qty': quantity, 'PnL': pnl, 'Result': 'Loss', 'Type': 'Stop Loss Hit'
                })
                position = 0
                
        elif position == -1: # Short
            if curr['Low'] <= target:
                pnl = (entry_price - target) * quantity
                
                # OPTION SIMULATION
                option_note = ""
                if ticker.startswith('^'):
                    pnl = pnl * 0.5
                    strike = get_atm_strike(entry_price, ticker)
                    option_note = f" [{strike} PE]"

                capital += pnl
                daily_pnl[current_date] = daily_pnl.get(current_date, 0) + pnl
                wins += 1
                trades.append({
                    'Time': timestamp, 'Symbol': ticker + option_note, 'Side': 'Short',
                    'Entry': entry_price, 'Exit': target, 
                    'Qty': quantity, 'PnL': pnl, 'Result': 'Win', 'Type': 'Target Hit'
                })
                position = 0
            elif curr['High'] >= stop_loss:
                pnl = (entry_price - stop_loss) * quantity
                
                # OPTION SIMULATION
                option_note = ""
                if ticker.startswith('^'):
                    pnl = pnl * 0.5
                    strike = get_atm_strike(entry_price, ticker)
                    option_note = f" [{strike} PE]"

                capital += pnl
                daily_pnl[current_date] = daily_pnl.get(current_date, 0) + pnl
                losses += 1
                trades.append({
                    'Time': timestamp, 'Symbol': ticker + option_note, 'Side': 'Short',
                    'Entry': entry_price, 'Exit': stop_loss, 
                    'Qty': quantity, 'PnL': pnl, 'Result': 'Loss', 'Type': 'Stop Loss Hit'
                })
                position = 0

    # Summary
    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    print("\n--- Backtest Results ---")
    print(f"Ticker: {ticker}")
    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Final Capital: {capital:.2f} (Start: {initial_capital})")
    print("------------------------")
    
    # Weekday Analysis
    weekday_stats = None
    if trades:
        df_t = pd.DataFrame(trades)
        df_t['Time'] = pd.to_datetime(df_t['Time'], utc=True)
        df_t['Weekday'] = df_t['Time'].dt.day_name()
        # Ensure order: Monday -> Friday
        weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        df_t['Weekday'] = pd.Categorical(df_t['Weekday'], categories=weekdays_order, ordered=True)
        
        stats = df_t.groupby('Weekday', observed=True).agg({
            'PnL': 'sum',
            'Result': 'count' # Total Trades
        }).rename(columns={'Result': 'Trades'})
        
        # Calculate Win Rate per day
        wins_per_day = df_t[df_t['Result'] == 'Win'].groupby('Weekday', observed=True).size()
        stats['Wins'] = wins_per_day
        stats['Wins'] = stats['Wins'].fillna(0)
        stats['Win Rate %'] = (stats['Wins'] / stats['Trades'] * 100).round(2)
        
        weekday_stats = stats[['Trades', 'PnL', 'Win Rate %']]

    return {
        'trades': trades,
        'win_rate': win_rate,
        'final_capital': capital,
        'data': intraday_df,
        'weekday_stats': weekday_stats
    }

if __name__ == "__main__":
    backtest_day_strategy("RELIANCE.NS")
    backtest_day_strategy("TATASTEEL.NS")
