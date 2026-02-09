"""
Test Live Execution Replay
--------------------------
Replays Today's Data (9:15 - 11:30) and:
1. Simulates Strategy State (PnL, Exit).
2. FIRES ACTUAL ORDERS to Fyers (to verify API calls).
"""

import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
import time
import logging
import os
from day_trading_strategy import fetch_data, calculate_9ema, check_3_candle_setup
from live_bot import execute_order, broker, get_strike_for_trade, get_fyers_symbol

# Config
TICKER = "^NSEI" # or ^NSEBANK
START_TIME = dt_time(9, 15)
END_TIME = dt_time(15, 29) # Full Day

def calculate_taxes(buy_val, sell_val):
    """
    Calculate Charges (Fyers/Zerodha Approx).
    """
    brokerage = 40 # Flat 20+20
    
    # STT: 0.1% on Sell Value (Options)
    stt = sell_val * 0.001 
    
    # Exchange Txn: ~0.05% on Turnover
    turnover = buy_val + sell_val
    exch_txn = turnover * 0.0005
    
    # GST: 18% on Brokerage + Exch Txn
    gst = (brokerage + exch_txn) * 0.18
    
    # Stamp Duty: 0.003% on Buy
    stamp = buy_val * 0.00003
    
    # SEBI: Negligible
    sebi = turnover * 0.000001
    
    total_tax = brokerage + stt + exch_txn + gst + stamp + sebi
    return total_tax

# Init Logger to Console
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ... (Process Logic)

def run_test():
    print(f"--- STARTING LIVE REPLAY TEST: {TICKER} (SURFER MODE - Last 7 Days) ---")
    
    # 1. Fetch Data
    daily, intraday = fetch_data(TICKER, broker=broker)
    if intraday.empty: return
    intraday = calculate_9ema(intraday)
    
    # Get Unique Dates
    all_dates = sorted(list(set(intraday.index.date)))
    # Last 7 days
    target_dates = all_dates[-7:]
    
    global trades_history
    trades_history = []
    qty = 65 
    
    for d in target_dates:
        d_str = d.strftime('%Y-%m-%d')
        today_data = intraday[intraday.index.date == d]
        if today_data.empty: continue
        
        print(f"\nProcessing {d_str} ...")
        
        position = 0
        entry_price = 0
        sl = 0
        
        full_idx = intraday.index.get_loc(today_data.index[0])
        
        for i in range(len(today_data)):
            curr_idx = full_idx + i
            curr_candle = intraday.iloc[curr_idx]
            ts = curr_candle.name.time()
            if ts > END_TIME: break
            
            # --- EXIT LOGIC (SURFER MODE) ---
            if position != 0:
                exit_reason = None
                
                # SURFER: Exit on 9EMA Crossover (Trend Reversal) or Hard SL
                ema = curr_candle['EMA9']
                
                if position == 1:
                    if curr_candle['Low'] <= sl: exit_reason = "SL HIT"
                    elif curr_candle['Close'] < ema: exit_reason = "SURFER EXIT (EMA X)" # Profit/Loss booking on trend change
                elif position == -1:
                    if curr_candle['High'] >= sl: exit_reason = "SL HIT"
                    elif curr_candle['Close'] > ema: exit_reason = "SURFER EXIT (EMA X)"
                
                if exit_reason:
                    spot_diff = (curr_candle['Close'] - entry_price) if position == 1 else (entry_price - curr_candle['Close'])
                    gross_pnl = spot_diff * qty * 0.7
                    
                    # Tax Calc
                    opt_buy_val = 200 * qty
                    opt_sell_val = (200 + (spot_diff*0.7)) * qty
                    taxes = calculate_taxes(opt_buy_val, opt_sell_val)
                    net_pnl = gross_pnl - taxes

                    trades_history.append({
                        'Date': d_str,
                        'Time': ts,
                        'Symbol': TICKER,
                        'Type': 'LONG' if position==1 else 'SHORT',
                        'Entry': entry_price,
                        'Exit': curr_candle['Close'],
                        'ExitReason': exit_reason,
                        'GrossPnL': gross_pnl,
                        'NetPnL': net_pnl
                    })
                    position = 0
                continue

            # --- ENTRY LOGIC ---
            if curr_idx < 4: continue
            df_slice = intraday.iloc[curr_idx-3 : curr_idx+1]
            sig = check_3_candle_setup(df_slice)
            
            if sig:
                # 3-Candle Setup
                signal_type = sig['Signal']
                entry_p = sig['Entry']
                sl_p = sig['SL']
                
                position = 1 if signal_type == 'LONG' else -1
                entry_price = entry_p
                sl = sl_p
                # No Target in Surfer Mode
                
    # --- REPORT ---
    print("\n" + "="*90)
    print("SURFER MODE RESULTS (Last 7 Days)")
    print("="*90)
    if trades_history:
        import pandas as pd
        df_res = pd.DataFrame(trades_history)
        print(df_res[['Date','Time','Type','Entry','Exit','ExitReason','NetPnL']].to_string(index=False))
        print("-" * 90)
        print(f"Total Net PnL: {df_res['NetPnL'].sum():.2f}")
        
        # Best Trade
        best = df_res.loc[df_res['NetPnL'].idxmax()]
        print(f"\n🚀 BEST SURFER TRADE: {best['Date']} {best['Type']} PnL: {best['NetPnL']:.2f}")
    else:
        print("No Trades.")
    print("="*90)

if __name__ == "__main__":
    run_test()
