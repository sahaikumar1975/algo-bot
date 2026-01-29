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

# List of Nifty 50 Stocks (Hardcoded for reliability if file missing)
NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LTIM.NS", "AXISBANK.NS", "LT.NS", "BAJFINANCE.NS", "MARUTI.NS",
    "ASIANPAINT.NS", "HCLTECH.NS", "TITAN.NS", "SUNPHARMA.NS", "TATASTEEL.NS",
    "NTPC.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "M&M.NS", "JSWSTEEL.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "TATAMOTORS.NS", "WIPRO.NS", "COALINDIA.NS",
    "ONGC.NS", "BAJAJFINSV.NS", "GRASIM.NS", "TECHM.NS", "HDFCLIFE.NS",
    "BRITANNIA.NS", "HEROMOTOCO.NS", "INDUSINDBANK.NS", "CIPLA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "UPL.NS", "SBILIFE.NS", "BAJAJ-AUTO.NS", "HINDALCO.NS",
    "^NSEI", "^NSEBANK"  # Added Indices
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

def scan_stock(ticker):
    """
    Analyzes a single stock and returns its status.
    """
    try:
        # 1. Fetch Daily Data (Last 5 days is enough for CPR)
        # 1. Fetch Daily Data (Last 5 days is enough for CPR)
        daily_df = download_with_retry(ticker, period='5d', interval='1d')
        if daily_df.empty or len(daily_df) < 2: return None
        
        # Clean columns
        if isinstance(daily_df.columns, pd.MultiIndex):
            daily_df.columns = daily_df.columns.get_level_values(0)
            
        # 2. Calculate CPR
        daily_cpr = calculate_cpr(daily_df)
        
        # Get Today's and Yesterday's CPR
        # daily_cpr index is the DATE the CPR is valid for
        # If running today (Live), we need CPR for Today.
        # Check last available index
        today_date = daily_cpr.index[-1].date()
        
        current_cpr = daily_cpr.iloc[-1]
        prev_cpr = daily_cpr.iloc[-2] # Previous day's CPR
        
        # 3. Analyze Trend (Bias)
        pivot = current_cpr['Pivot']
        prev_pivot = prev_cpr['Pivot']
        
        trend = "NEUTRAL"
        if pivot > prev_pivot: trend = "BULLISH"
        elif pivot < prev_pivot: trend = "BEARISH"
        
        # 4. Narrow CPR Check
        cpr_width = abs(current_cpr['TC'] - current_cpr['BC'])
        is_narrow = cpr_width < (pivot * 0.005) # 0.5% Width
        
        # 5. Fetch Intraday Data (Current state)
        # Fetch last 5 candles to check current price status
        # 5. Fetch Intraday Data (Current state)
        # Fetch last 5 candles to check current price status
        intraday = download_with_retry(ticker, period='1d', interval='5m')
        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)
            
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

def run_scanner():
    print(f"--- Starting Market Scanner ({len(NIFTY_50)} Stocks) ---")
    print("Scanning for CPR Trends and Intraday Alignments...")
    
    results = []
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(scan_stock, t): t for t in NIFTY_50}
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
