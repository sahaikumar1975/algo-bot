
import yfinance as yf
import pandas as pd
from day_trading_strategy import calculate_9ema

def debug_ongc():
    ticker = "JINDALSTEL.NS"
    print(f"Fetching data for {ticker}...")
    
    # Fetch data for today
    df = yf.download(ticker, period="5d", interval="5m", progress=False)
    
    if df.empty:
        print("No data found.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = calculate_9ema(df)
    df['TimeStr'] = df.index.astype(str)
    
    # Filter for Monday 9th Feb early morning
    # 03:45 UTC = 9:15 AM IST
    # 04:45 UTC = 10:15 AM IST
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    day_data = df[df.index.strftime('%Y-%m-%d') == '2026-02-09']
    mon_data = day_data.between_time('03:45', '04:45')
    
    print(f"\n--- {ticker} Early Data (2026-02-09) ---")
    print(mon_data[['Open', 'High', 'Low', 'Close', 'EMA9']].to_string())

if __name__ == "__main__":
    debug_ongc()
