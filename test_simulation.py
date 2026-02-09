import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Setup Logging to Console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("--- STARTING 9EMA SIMULATION TEST ---")
print("1. Mocking Fyers connection (Paper Trading Mode)")
print("2. Mocking Market Data (Forcing a 9EMA BUY Setup)")
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
def mock_fetch_data(ticker, broker=None):
    print(f"   -> Mocking Data Fetch for {ticker}...")
    
    # Create Dummy Dates
    dates_daily = pd.date_range(end=datetime.now().date(), periods=5)
    dates_intra = pd.date_range(end=datetime.now(), periods=50, freq='5min')
    
    # DAILY DATA
    daily_df = pd.DataFrame({
        'Open': [100]*5, 'High': [105]*5, 'Low': [95]*5, 'Close': [101]*5, 'Volume': [1000]*5
    }, index=dates_daily)
    
    # INTRADAY DATA (For 9EMA Signal)
    # Scenario: 9EMA BUY SETUP
    # We need EMA9 to be well above the Alert Candle High.
    
    df = pd.DataFrame(index=dates_intra)
    
    # Generate a Downtrend so EMA is high
    # Prices dropping from 110 to 102
    prices = np.linspace(110, 102, 48).tolist()
    
    # Candle -2 (Alert Candle):
    # Needs High < EMA9.
    # If EMA is trailing around 102+, let's make Alert Candle Low.
    prices.append(100.0) # Index -2 close
    
    # Candle -1 (Trigger Candle):
    # Needs to break High of Alert.
    prices.append(101.0) # Index -1 close
    
    df['Close'] = prices
    df['Open'] = prices 
    df['High'] = [p + 0.5 for p in prices] # Wicks
    df['Low'] = [p - 0.5 for p in prices]
    df['Volume'] = 10000 
    
    # FORCE SPECIFICS
    
    # Alert Candle (Index -2)
    # High = 100.5
    # Low = 99.5
    # Close = 100.0
    # EMA should be > 100.5. (Likely is, given previous prices 102+)
    df.iloc[-2, df.columns.get_loc('High')] = 100.5
    df.iloc[-2, df.columns.get_loc('Low')] = 99.5
    df.iloc[-2, df.columns.get_loc('Close')] = 100.0
    
    # Trigger Candle (Index -1)
    # High = 101.5 (Breaks 100.5)
    # Low = 100.8
    # Close = 101.0
    df.iloc[-1, df.columns.get_loc('High')] = 101.5
    df.iloc[-1, df.columns.get_loc('Low')] = 100.8
    df.iloc[-1, df.columns.get_loc('Close')] = 101.0
    
    return daily_df, df

# 3. Patch the bot
live_bot.fetch_data = mock_fetch_data

# --- RUN TEST ---

# Create a Watchlist with a Mock Stock
watchlist = [
    {'ticker': 'SIM_STOCK', 'trend': 'BULLISH'},
    {'ticker': '^NSEI', 'trend': 'NEUTRAL'} 
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
        print(df[['Time', 'Ticker', 'Signal', 'Entry_Price', 'SL', 'Target']].to_string())
    else:
        print("❌ Trade Log exists but is empty. Signal Logic didn't trigger?")
else:
    print("❌ No Trade Log created. Signal Logic didn't trigger.")

print("\nDone.")
