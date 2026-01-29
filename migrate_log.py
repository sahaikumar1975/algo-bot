import pandas as pd
import os

TRADE_LOG = "/Users/sahaikumar/Projects/SMA2150/trade_log.csv"

TRADE_COLUMNS = ['Time', 'Ticker', 'Signal', 'Entry_Price', 'Qty', 'SL', 'Target', 'Notes', 'Exit_Time', 'Exit_Price', 'Exit_Reason', 'PnL', 'Status']

if os.path.exists(TRADE_LOG):
    try:
        df = pd.read_csv(TRADE_LOG)
        print(f"Original columns: {df.columns.tolist()}")
        
        # Migration Logic
        # 1. Rename 'Price' to 'Entry_Price' if it exists
        if 'Price' in df.columns and 'Entry_Price' not in df.columns:
            df.rename(columns={'Price': 'Entry_Price'}, inplace=True)
            
        # 2. Add missing columns
        for col in TRADE_COLUMNS:
            if col not in df.columns:
                df[col] = None
                
        # 3. Fill Status for old trades
        if 'Status' in df.columns:
            df['Status'].fillna('OPEN', inplace=True)
            
        # Reorder
        df = df[TRADE_COLUMNS]
        
        # Save back
        df.to_csv(TRADE_LOG, index=False)
        print("Migration successful. file saved.")
        
    except Exception as e:
        print(f"Migration failed: {e}")
else:
    print("No trade_log.csv found.")
