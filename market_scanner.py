"""
Market Scanner for Day Trading Robot
Scans Nifty 50/500 stocks for:
1. Daily CPR Trend (Bullish/Bearish)
2. Narrow CPR (Trend Day Probability)
3. Live Intraday Conditions (Price vs VWAP, RSI)
"""

import pandas as pd
import yfinance as yf
import numpy as np
import concurrent.futures
from datetime import datetime, timedelta
import os
import sys
import time

# Import logic from strategy file
from day_trading_strategy import calculate_cpr, add_rsi

# Optimized Watchlist (Backtested Profitable on 9EMA Strict Strategy)
# Filtered from Top 60 Stocks based on >40% Win Rate & Positive PnL
NIFTY_50 = [
    'AMBUJACEM.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BRITANNIA.NS', 'CHOLAFIN.NS', 
    'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'HAVELLS.NS', 'HDFCBANK.NS', 
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'ICICIBANK.NS', 'INFY.NS', 'JINDALSTEL.NS', 
    'LT.NS', 'LTIM.NS', 'MARUTI.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 
    'SBICARD.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SIEMENS.NS',
    '^NSEI', '^NSEBANK'  # Indices
]

def download_with_retry(ticker, period, interval, retries=3):
    """
    Downloads data with retry mechanism for transient network errors.
    """
    last_exception = None
    for i in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            return df
        except Exception as e:
            last_exception = e
            if i < retries - 1:
                sleep_time = 2 * (i + 1)
                # print(f"⚠️ Network error for {ticker}. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            
    # If all retries fail, re-raise the last exception
    if last_exception:
        raise last_exception
    return pd.DataFrame()

def scan_stock(ticker, broker=None):
    """
    Analyzes a single stock and returns its status.
    """
    try:
        # 1. Fetch Daily Data (Last 5 days is enough for CPR)
        # 1. Fetch Daily Data (Last 5 days is enough for CPR)
        # Using shared fetch_data which handles retries and timeouts
        from day_trading_strategy import fetch_data
        
        # fetch_data returns (daily, intraday)
        # We just need daily here first, but might as well get both if optimized, 
        # or we can modify scan_stock to use the tuple directly.
        # However, scan_stock logic is split. Let's see.
        
        # Actually scanner calls download twice with different params. 
        # fetch_data does both in one go. Let's maximize efficiency.
        
        daily_df, intraday_int_df = fetch_data(ticker, broker=broker)
        
        if daily_df.empty or len(daily_df) < 2: return None
        
        # Clean columns
        if isinstance(daily_df.columns, pd.MultiIndex):
            daily_df.columns = daily_df.columns.get_level_values(0)
            
        # 2. Calculate CPR (Live Mode)
        # We need CPR for TODAY (based on Yesterday's Data)
        # daily_df.iloc[-1] is Yesterday (if pre-market)
        
        last_candle = daily_df.iloc[-1]
        prev_candle = daily_df.iloc[-2]
        
        # Calculate PIVOT for TODAY (Using Last Candle)
        pivot = (last_candle['High'] + last_candle['Low'] + last_candle['Close']) / 3
        bc = (last_candle['High'] + last_candle['Low']) / 2
        tc = (pivot - bc) + pivot
        
        # Calculate PIVOT for YESTERDAY (Using Prev Candle)
        prev_pivot_val = (prev_candle['High'] + prev_candle['Low'] + prev_candle['Close']) / 3
        
        # 3. Analyze Trend (Bias)
        # Compare Today's Pivot vs Yesterday's Pivot
        trend = "NEUTRAL"
        if pivot > prev_pivot_val: trend = "BULLISH"
        elif pivot < prev_pivot_val: trend = "BEARISH"
        
        # 4. Narrow CPR Check
        cpr_width = abs(tc - bc)
        is_narrow = cpr_width < (pivot * 0.005) # 0.5% Width
        
        # 5. Fetch Intraday Data (Current state)
        # Reuse data from fetch_data call above
        intraday = intraday_int_df
            
        if intraday.empty:
            return {
                'Ticker': ticker, 'Trend': trend, 'Narrow_CPR': is_narrow, 
                'Price': 'N/A', 'Status': 'No Intraday Data'
            }
            
        last_candle = intraday.iloc[-1]
        current_price = last_candle['Close']
        
        # Calculate VWAP
        cum_vol = intraday['Volume'].cumsum()
        cum_vol_price = (intraday['Close'] * intraday['Volume']).cumsum()
        vwap = cum_vol_price / cum_vol
        current_vwap = vwap.iloc[-1]
        
        # Check alignment
        status = "WAIT"
        
        if trend == "BULLISH" and current_price > current_vwap:
            status = "LOOKING FOR LONG"
            if is_narrow: status += " (High Prob)"
            
        elif trend == "BEARISH" and current_price < current_vwap:
            status = "LOOKING FOR SHORT"
            if is_narrow: status += " (High Prob)"
            
        return {
            'Ticker': ticker,
            'Trend': trend,
            'Narrow_CPR': is_narrow,
            'Price': round(current_price, 2),
            'VWAP': round(current_vwap, 2),
            'Status': status
        }

    except Exception as e:
        print(f"Error scanning {ticker}: {e}")
        return None

def run_scanner(broker=None):
    print(f"--- Starting Market Scanner ({len(NIFTY_50)} Stocks) ---")
    print("Scanning for CPR Trends and Intraday Alignments...")
    
    results = []
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(scan_stock, t, broker): t for t in NIFTY_50}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
    
    # Convert to DataFrame for display
    df_res = pd.DataFrame(results)
    
    # Filter for 'High Prob' or Active setups
    if not df_res.empty:
        # Sort by Status interestingness
        df_res.sort_values(by=['Status', 'Trend'], ascending=False, inplace=True)
        
        print("\n--- Market Scan Results ---")
        print(df_res.to_string(index=False))
        
        # Save to CSV
        df_res.to_csv("daily_scan_results.csv", index=False)
        print("\nResults saved to daily_scan_results.csv")
        print("\nResults saved to daily_scan_results.csv")
        return results # Return list of dicts for bot
    else:
        print("No results found.")
        return []

if __name__ == "__main__":
    run_scanner()
