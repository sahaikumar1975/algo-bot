
import yfinance as yf
import pandas as pd
from day_trading_strategy import calculate_9ema

def debug_heromotoco():
    ticker = "HEROMOTOCO.NS"
    print(f"Fetching data for {ticker}...")
    
    # Get 2 days of 5m data
    df = yf.download(ticker, period="5d", interval="5m", progress=False)
    
    if df.empty:
        print("No data found.")
        return

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate 9EMA
    df = calculate_9ema(df)

    # Get last 10 candles
    print("\n--- Last 10 Candles (including today's open) ---")
    print(df[['Open', 'High', 'Low', 'Close', 'EMA9']].tail(10))

    # Check setup manually for the last 4 candles ending at first candle of today
    # Assuming the trade was at 9:15:23, that's the first candle of today (index -1 if downloaded now)
    # Actually if run now (mid-day), today has many candles.
    # We want to see the transition from Friday Close to Monday Open.
    
    # Filter for Friday Close and Monday Open
    # Friday 6th Feb to Monday 9th Feb
    
    mask = (df.index >= "2026-02-06 14:00:00") & (df.index <= "2026-02-09 10:00:00")
    subset = df.loc[mask]
    
    print("\n--- Transition Period Data (Fri Close -> Mon Open) ---")
    print(subset[['Open', 'High', 'Low', 'Close', 'EMA9']].to_string())

if __name__ == "__main__":
    debug_heromotoco()
