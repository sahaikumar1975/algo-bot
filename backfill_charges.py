import pandas as pd
import os

TRADE_LOG = "trade_log.csv"

def calculate_charges(entry_price, exit_price, qty, ticker, is_option):
    try:
        entry_price = float(entry_price)
        exit_price = float(exit_price)
        qty = float(qty)
    except:
        return 0.0

    # Turnover
    entry_turnover = entry_price * qty
    exit_turnover = exit_price * qty
    total_turnover = entry_turnover + exit_turnover
    
    # 1. Brokerage (Fyers: Max 20 per order)
    brokerage_entry = 20.0
    brokerage_exit = 20.0
    
    
    # If not option (Equity Intraday)
    if not is_option:
        # Equity Intraday: Min(20, 0.03%)
        brokerage_entry = min(20.0, entry_turnover * 0.0003)
        brokerage_exit = min(20.0, exit_turnover * 0.0003)
    else:
        # OPTION CHECK: If Entry Price > 5000, it's likely Index Spot Price (Paper Trade)
        # We must normalize it to a realistic Premium (~0.6% for Nifty, ~0.8% for BankNifty)
        if entry_price > 5000:
            if 'BANK' in ticker.upper(): entry_price = entry_price * 0.008
            else: entry_price = entry_price * 0.006

        if exit_price > 5000:
            if 'BANK' in ticker.upper(): exit_price = exit_price * 0.008
            else: exit_price = exit_price * 0.006
            
        # Recalculate Turnover with normalized prices
        entry_turnover = entry_price * qty
        exit_turnover = exit_price * qty
        total_turnover = entry_turnover + exit_turnover

        
    total_brokerage = brokerage_entry + brokerage_exit
    
    # 2. STT (Security Transaction Tax)
    # Equity Intraday: 0.025% on SELL only
    # Options: 0.0625% on SELL only (on premium)
    stt = 0.0
    if is_option:
        stt = exit_turnover * 0.000625
    else:
        stt = exit_turnover * 0.00025
        
    # 3. Transaction Charges (NSE)
    # NSE Equity: 0.00325%
    # NSE Options: 0.05% (on premium)
    # Note: 0.05% = 0.0005
    txn_rate = 0.0005 if is_option else 0.0000325
    txn_charges = total_turnover * txn_rate
    
    # 4. GST (18% on Brokerage + Txn Charges)
    gst = (total_brokerage + txn_charges) * 0.18
    
    # 5. SEBI Charges (10 per crore = 0.0001%)
    # 0.0001% = 0.000001
    sebi_charges = total_turnover * 0.000001
    
    # 6. Stamp Duty (0.003% on Buy only)
    # 0.003% = 0.00003
    stamp_duty = entry_turnover * 0.00003
    
    total_charges = total_brokerage + stt + txn_charges + gst + sebi_charges + stamp_duty
    return round(total_charges, 2)

if os.path.exists(TRADE_LOG):
    df = pd.read_csv(TRADE_LOG)
    print("Backfilling Charges...")
    
    count = 0
    for idx, row in df.iterrows():
        status = str(row['Status']).upper()
        
        # Only calculate for CLOSED trades
        if 'CLOSED' in status:
            ticker = str(row['Ticker'])
            instrument = str(row['Instrument'])
            is_option = "CE" in instrument or "PE" in instrument
            
            entry = row['Entry_Price']
            exit_p = row['Exit_Price']
            qty = row['Qty']
            pnl = row['PnL']
            
            try: pnl = float(pnl)
            except: pnl = 0.0
            
            charges = calculate_charges(entry, exit_p, qty, ticker, is_option)
            net_pnl = pnl - charges
            
            df.at[idx, 'Charges'] = charges
            df.at[idx, 'Net_PnL'] = net_pnl
            count += 1
            
    if count > 0:
        df.to_csv(TRADE_LOG, index=False)
        print(f"✅ Updated charges for {count} trades.")
    else:
        print("No trades needed backfilling.")
        
else:
    print("Trade log not found.")
