
import yfinance as yf
import pandas as pd
from day_trading_strategy import calculate_9ema

def debug_heromotoco():
    ticker = "HEROMOTOCO.NS"
    # Fetch 5 days of data to ensure we get Friday and Monday
    df = yf.download(ticker, period="5d", interval="5m", progress=False)
    
    if df.empty:
        print("No data found.")
        return

    # Flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate 9EMA
    df = calculate_9ema(df)

    # Filter for Friday (6th) late hours and Monday (9th) early hours
    # Timestamps are likely timezone-aware (IST usually for NSE if yfinance setup allows, or UTC)
    # Let's convert to string to filter safely or just use tail if we know the count.
    
    # Check the index last val to see timezone
    # print(f"Timezone info: {df.index.dtype}")
    
    # We want last 5 candles of Friday and first 3 of Monday
    # Friday close is index where date is 2026-02-06 and time is 15:25 (start of last candle)
    # Monday open is 2026-02-09 09:15.
    
    # Create string column for easier filtering
    df['TimeStr'] = df.index.astype(str)
    
    fri_mask = df['TimeStr'].str.contains('2026-02-06')
    mon_mask = df['TimeStr'].str.contains('2026-02-09')
    
    fri_data = df[fri_mask].tail(5)
    mon_data = df[mon_mask].head(5)
    
    print("\n--- Friday Close Data (Potential Setup) ---")
    print(fri_data[['Open', 'High', 'Low', 'Close', 'EMA9']].to_string())
    
    print("\n--- Monday Open Data (Entry Trigger) ---")
    print(mon_data[['Open', 'High', 'Low', 'Close', 'EMA9']].to_string())

if __name__ == "__main__":
    debug_heromotoco()
