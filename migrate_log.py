import pandas as pd
import os

TRADE_LOG = "trade_log.csv"
NEW_COLS = ['Charges', 'Net_PnL']

if os.path.exists(TRADE_LOG):
    df = pd.read_csv(TRADE_LOG)
    print("Old Columns:", df.columns.tolist())
    
    migrated = False
    for col in NEW_COLS:
        if col not in df.columns:
            df[col] = 0.0
            migrated = True
            
    if migrated:
        df.to_csv(TRADE_LOG, index=False)
        print("✅ MIGRATION SUCCESS! Added Charges/Net_PnL columns.")
        print("New Columns:", df.columns.tolist())
    else:
        print("No migration needed.")
else:
    print("Trade Log not found.")
