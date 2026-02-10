import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

def check_data():
    ticker = "^NSEI"
    df = yf.download(ticker, period="5d", interval="5m", progress=False)
    
    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calculate 9EMA
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    
    # Convert index to IST for readability
    df.index = df.index.tz_convert('Asia/Kolkata')
    
    # Filter 11:00 to 11:45 IST today
    target_date = "2026-02-10"
    df['TimeStr'] = df.index.strftime("%Y-%m-%d %H:%M:%S")
    
    # Filter today
    df_today = df[df['TimeStr'].str.contains(target_date)]
    
    # Filter range 11:00 to 11:45
    # Just look for 11:00, 11:05 ... 11:45
    df_final = df_today[
        (df_today['TimeStr'].str.contains(" 11:0")) | 
        (df_today['TimeStr'].str.contains(" 11:1")) |
        (df_today['TimeStr'].str.contains(" 11:2")) |
        (df_today['TimeStr'].str.contains(" 11:3")) |
        (df_today['TimeStr'].str.contains(" 11:4"))
    ]
    
    print(f"\nData for {ticker} (IST):")
    for index, row in df_final.iterrows():
        t_str = row['TimeStr']
        o = row['Open']
        c = row['Close']
        ema = row['EMA9']
        
        try: o = float(o)
        except: pass
        try: c = float(c)
        except: pass
        try: ema = float(ema)
        except: pass
        
        above = c > ema
        open_above = o > ema
        
        # Check Cross Logic
        # Short Cross: Open > EMA and Close < EMA
        is_cross_down = open_above and not above
        
        print(f"{t_str} | Open: {o:.2f} | Close: {c:.2f} | EMA9: {ema:.2f} | Close<EMA? {not above} | Open>EMA? {open_above} | CROSS DOWN? {is_cross_down}")

check_data()
