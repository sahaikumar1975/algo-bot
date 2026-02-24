import pandas as pd
import yfinance as yf
from day_trading_strategy import calculate_ema, check_orb_breakout_setup
import numpy as np
import warnings
import sys

warnings.filterwarnings('ignore')

OPTIONS_STOCKS = ["BHARTIARTL", "TCS", "LT"]

def backtest_options_strategy(ticker, test_days=60):
    """
    Backtest 15m ORB with Options Buying Simulation.
    Assumes ATM Option (Delta 0.5)
    """
    print(f"\n--- Options Buying Backtest: {ticker} (Simulator) ---")
    ticker_yf = f"{ticker}.NS" if not ticker.endswith('.NS') else ticker
    
    # Fetch 15m Spot data
    df = yf.download(ticker_yf, period=f"{test_days}d", interval="15m", progress=False)
    if df.empty: return []
    
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df = calculate_ema(df, period=9) # Using 9EMA for options trailing
    
    if df.index.tz is None: df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else: df.index = df.index.tz_convert('Asia/Kolkata')
        
    test_dates = np.unique([d.date() for d in df.index])
    trades = []
    
    # Simulation Parameters
    DELTA = 0.55 # Slightly aggressive for ATM
    THETA_PCT = 0.005 # 0.5% decay of price per day (simplified)
    
    for current_date in test_dates:
        today_data = df[df.index.date == current_date]
        if len(today_data) < 10: continue
        
        position = 0 # 1 for CE, -1 for PE
        entry_price = 0
        sl = 0
        entry_time = None
        
        # Established 15m Range (9:15-9:30)
        # Note: check_orb_breakout_setup handles range logic
        
        for i in range(1, len(today_data)):
            curr = today_data.iloc[i]
            ts = today_data.index[i]
            
            if position != 0:
                exit_reason = None
                spot_close = curr['Close']
                
                # Exit Conditions
                if position == 1: # LONG (CE Buy)
                    if curr['Low'] <= sl: exit_reason = "Spot SL Hit"
                    elif spot_close < curr['EMA9']: exit_reason = "9EMA Trail"
                elif position == -1: # SHORT (PE Buy)
                    if curr['High'] >= sl: exit_reason = "Spot SL Hit"
                    elif spot_close > curr['EMA9']: exit_reason = "9EMA Trail"
                
                if ts.hour >= 15 and ts.minute >= 15: exit_reason = "EOD"
                
                if exit_reason:
                    spot_exit = sl if exit_reason == "Spot SL Hit" else spot_close
                    spot_pnl = (spot_exit - entry_price) * position
                    
                    # SIMULATED OPTIONS PNL
                    # Option PnL = (Spot Change * Delta)
                    option_pnl = spot_pnl * DELTA
                    
                    trades.append({
                        'Date': current_date,
                        'Time': entry_time.time(),
                        'Type': 'CE' if position == 1 else 'PE',
                        'Spot_Entry': round(entry_price, 2),
                        'Spot_Exit': round(spot_exit, 2),
                        'Exit_Reason': exit_reason,
                        'Option_PnL_Pts': round(option_pnl, 2)
                    })
                    position = 0
                    continue

            if position == 0 and ts.hour < 15:
                # Supply current slice
                signal_data = check_orb_breakout_setup(today_data.iloc[:i+1])
                if signal_data:
                    position = 1 if signal_data['Signal'] == 'LONG' else -1
                    entry_price = signal_data['Entry']
                    sl = signal_data['SL']
                    entry_time = ts
                    
    if not trades:
        print("No trades triggered.")
        return []
        
    res_df = pd.DataFrame(trades)
    wins = len(res_df[res_df['Option_PnL_Pts'] > 0])
    total = len(res_df)
    win_rate = (wins/total)*100
    total_opt_pnl = res_df['Option_PnL_Pts'].sum()
    
    print(f"Total Trades: {total}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Net Option Points (Simulated): {total_opt_pnl:.2f}")
    print(res_df[['Date', 'Type', 'Exit_Reason', 'Option_PnL_Pts']].to_string(index=False))
    
    return trades

def run_suite():
    print("=== OPTIONS BUYING BACKTEST SUITE (TOP 3 LIQUID) ===")
    all_summary = []
    for ticker in OPTIONS_STOCKS:
        trades = backtest_options_strategy(ticker)
        if trades:
            pnl = sum([t['Option_PnL_Pts'] for t in trades])
            all_summary.append({'Ticker': ticker, 'PnL': pnl, 'Trades': len(trades)})
            
    print("\n" + "="*40)
    print("FINAL OPTIONS SUMMARY (60 DAYS)")
    print("="*40)
    print(pd.DataFrame(all_summary).to_string(index=False))

if __name__ == "__main__":
    run_suite()
