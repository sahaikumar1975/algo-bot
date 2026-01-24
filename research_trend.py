
from day_trading_strategy import fetch_data, calculate_cpr, add_rsi
import pandas as pd
import numpy as np

def backtest_trend_following(ticker, capital=100000):
    daily, df = fetch_data(ticker)
    
    # Indicators
    daily_cpr = calculate_cpr(daily)
    df = add_rsi(df)
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # ATR
    high_low = df['High'] - df['Low']
    h_c = (df['High'] - df['Close'].shift()).abs()
    l_c = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, h_c, l_c], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    # VWAP
    df['Cum_Vol'] = df.groupby(df.index.date)['Volume'].cumsum()
    df['Cum_Vol_Price'] = df.groupby(df.index.date).apply(lambda x: (x['Close'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
    df['VWAP'] = df['Cum_Vol_Price'] / df['Cum_Vol']
    
    # Merge CPR
    df['Date_Only'] = df.index.date
    daily_cpr['Date_Only'] = daily_cpr.index.date
    daily_cpr['Prev_Pivot'] = daily_cpr['Pivot'].shift(1) # Add Prev Pivot
    df = df.merge(daily_cpr[['Pivot', 'TC', 'BC', 'Prev_Pivot', 'Date_Only']], on='Date_Only', how='left')
    df.index = pd.to_datetime(df.index) # Restore Index
    
    trades = []
    position = 0 # 1 Long, -1 Short
    entry_price = 0
    qty = 0
    wins = 0
    losses = 0
    
    for i in range(20, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Timezone Fix
        timestamp = curr.name
        if timestamp.tzinfo is not None:
             timestamp = timestamp.tz_convert('Asia/Kolkata')
        
        # Time Filter
        if timestamp.time() >= pd.Timestamp("15:00").time():
            if position != 0:
                pnl = (curr['Close'] - entry_price) * qty * position
                if ticker.startswith('^'): pnl *= 0.5
                capital += pnl
                trades.append(pnl)
                position = 0
            continue
            
        if timestamp.time() < pd.Timestamp("09:30").time(): continue

        # Entry Logic (Same as Strategy)
        upper = max(curr['TC'], curr['BC'])
        lower = min(curr['TC'], curr['BC'])
        narrow = abs(curr['TC'] - curr['BC']) < (curr['Close'] * 0.0025)
        
        if position == 0:
            # Long
            if (curr['Pivot'] > curr['Prev_Pivot']) and (curr['Close'] > curr['VWAP']):
                if (prev['Close'] < upper and curr['Close'] > upper) or (prev['RSI'] <= 60 and curr['RSI'] > 60):
                    if narrow:
                        position = 1
                        entry_price = curr['Close']
                        # Lot Size
                        lot_size = 30 if 'BANK' in ticker else 65
                        qty = lot_size # Static 1 Lot for test
                        continue

            # Short
            if (curr['Pivot'] < curr['Prev_Pivot']) and (curr['Close'] < curr['VWAP']):
                if (prev['Close'] > lower and curr['Close'] < lower) or (prev['RSI'] >= 40 and curr['RSI'] < 40):
                    if narrow:
                        position = -1
                        entry_price = curr['Close']
                        lot_size = 30 if 'BANK' in ticker else 65
                        qty = lot_size
                        continue
        
        # Exit Logic: TRAILING STOP (EMA 20 Cross)
        elif position == 1:
            # Exit if Close < EMA 20
            if curr['Close'] < curr['EMA20']:
                pnl = (curr['Close'] - entry_price) * qty
                if ticker.startswith('^'): pnl *= 0.5
                capital += pnl
                trades.append(pnl)
                position = 0
                
        elif position == -1:
            # Exit if Close > EMA 20
            if curr['Close'] > curr['EMA20']:
                pnl = (entry_price - curr['Close']) * qty
                if ticker.startswith('^'): pnl *= 0.5
                capital += pnl
                trades.append(pnl)
                position = 0

    return capital, len(trades)

print("--- TREND FOLLOWING TEST (Exit on EMA 20 Cross) ---")
cap_nifty, tr_nifty = backtest_trend_following('^NSEI')
print(f"NIFTY: Trades: {tr_nifty} | Final: {cap_nifty:.2f}")

cap_bank, tr_bank = backtest_trend_following('^NSEBANK')
print(f"BANKNIFTY: Trades: {tr_bank} | Final: {cap_bank:.2f}")
