import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Setup Logging to Console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("--- STARTING SIMULATION TEST ---")
print("1. Mocking Fyers connection (Paper Trading Mode)")
print("2. Mocking Market Data (Forcing a BULLISH Setup)")
print("3. Checking if Signal triggers AI and Logging")

# Import the bot module
try:
    import live_bot
except ImportError as e:
    print(f"❌ Failed to import live_bot: {e}")
    exit(1)

# --- MOCKING ---

# 1. Force Paper Trade
live_bot.LIVE_TRADE = False
live_bot.broker = None
live_bot.TRADE_LOG = "simulation_trade_log.csv" # Separate log file

# 2. Mock Data Generator
def mock_fetch_data(ticker):
    print(f"   -> Mocking Data Fetch for {ticker}...")
    
    # Create Dummy Dates
    dates_daily = pd.date_range(end=datetime.now().date(), periods=5)
    dates_intra = pd.date_range(end=datetime.now(), periods=50, freq='5min')
    
    # DAILY DATA (For CPR)
    # Pivot = (H+L+C)/3
    # We want Narrow CPR? Not strictly checked in check_for_signals logic (it uses 'trend' bias passed in), 
    # but let's make it normal.
    daily_df = pd.DataFrame({
        'Open': [100]*5, 'High': [105]*5, 'Low': [95]*5, 'Close': [101]*5, 'Volume': [1000]*5
    }, index=dates_daily)
    
    # INTRADAY DATA (For Signal)
    # Scenario: BULLISH
    # Price > VWAP
    # RSI > 60 (Cross)
    # Volume > SMA20
    
    df = pd.DataFrame(index=dates_intra)
    df['Open'] = 100.0
    df['High'] = 102.0
    df['Low'] = 99.0
    df['Close'] = 101.0 # Will be Price
    df['Volume'] = 10000 # Base Volume
    
    
    # Generate prices: Flat then SUDDEN Spike to trigger RSI Cross > 60 at the very end
    # Flat 100 for 49 candles -> RSI ~50
    # Then 100 -> 105 spike -> RSI jumps > 60
    
    prices = [100.0] * 48 + [100.0, 105.0] # 50 candles
    df['Close'] = prices
    df['High'] = [p + 2.0 for p in prices] # Ensure High is higher for CPR logic
    df['Low'] = [p - 2.0 for p in prices]
    
    # SPIKE VOLUME on the last candle to satisfy Vol > SMA20
    df.iloc[-1, df.columns.get_loc('Volume')] = 50000 # 5x average
    
    return daily_df, df

# 3. Patch the bot
live_bot.fetch_data = mock_fetch_data

# --- RUN TEST ---

# Create a Watchlist with a Mock Stock
watchlist = [
    {'ticker': 'SIM_STOCK', 'trend': 'BULLISH'},
    {'ticker': '^NSEI', 'trend': 'NEUTRAL'} # Index to ignore or test
]

print("\n--- RUNNING CHECK_FOR_SIGNALS ---")
try:
    live_bot.check_for_signals(watchlist)
    print("\n✅ Simulation Logic Completed.")
except Exception as e:
    print(f"\n❌ Simulation Crashed: {e}")
    import traceback
    traceback.print_exc()

# --- VALIDATE OUTPUT ---

print("\n--- RESULTS ---")
if os.path.exists("simulation_trade_log.csv"):
    df = pd.read_csv("simulation_trade_log.csv")
    if not df.empty:
        print(f"✅ Trade Log Created! Found {len(df)} trades.")
        print(df[['Time', 'Ticker', 'Signal', 'Entry_Price', 'Notes']].tail())
        
        # Check if AI was called
        last_note = df.iloc[-1]['Notes']
        if "[SNIPER]" in last_note or "[SURFER]" in last_note:
            print("✅ AI Validation Success (Mode returned).")
        else:
            print(f"⚠️  AI Response might be missing/generic: {last_note}")
            
    else:
        print("❌ Trade Log exists but is empty. Signal Logic didn't trigger?")
else:
    print("❌ No Trade Log created. Signal Logic didn't trigger.")

print("\nDone.")
