
import pandas as pd
from day_trading_strategy import backtest_day_strategy
import sys

def main():
    print("--- Single Stock Backtest (9EMA Strict) ---")
    ticker = input("Enter Stock Ticker (e.g., RELIANCE): ").strip().upper()
    if not ticker.endswith('.NS') and not ticker.startswith('^'):
        ticker += '.NS'
        
    mode_input = input("Enter Strategy Mode (1=SNIPER, 2=SURFER) [Default: SNIPER]: ").strip()
    mode = 'SURFER' if mode_input == '2' else 'SNIPER'
        
    print(f"Fetching data and running backtest for {ticker} in {mode} Mode...")
    try:
        trades = backtest_day_strategy(ticker, strategy_mode=mode)
        
        if not trades:
            print("No trades found or no data available.")
            return

        df = pd.DataFrame(trades)
        
        # Summary Stats
        total_trades = len(df)
        total_pnl = df['PnL'].sum()
        win_trades = len(df[df['PnL'] > 0])
        loss_trades = len(df[df['PnL'] <= 0])
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        print("\n--- Backtest Results ---")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}% ({win_trades}W / {loss_trades}L)")
        print(f"Total PnL: ₹{total_pnl:.2f}")
        
        print("\n--- Trade Log ---")
        # Reorder columns
        cols = ['Entry Time', 'Exit Time', 'Signal', 'Entry', 'Exit', 'PnL', 'Reason']
        print(df[cols].to_string(index=False))
        
        # Save
        filename = f"backtest_{ticker.replace('.NS','')}.csv"
        df.to_csv(filename, index=False)
        print(f"\nDetailed log saved to {filename}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
