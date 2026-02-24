from backtest_orb_hmm import backtest_orb_hmm_strategy
import pandas as pd
import os

ELITE_ORB_BUNDLE = [
    "MARUTI", "APOLLOHOSP", "LTIM", "SIEMENS", 
    "ASIANPAINT", "TVSMOTOR", "BHARTIARTL", 
    "HINDUNILVR", "TCS", "DLF"
]

def run_suite():
    print("=== ELITE ORB BUNDLE BACKTEST SUITE ===")
    print(f"Backtesting {len(ELITE_ORB_BUNDLE)} stocks for the last 60 days (15m ORB Strategy)...")
    
    all_results = []
    
    for ticker in ELITE_ORB_BUNDLE:
        try:
            trades = backtest_orb_hmm_strategy(ticker, start_days=180, test_days=60)
            if trades:
                df = pd.DataFrame(trades)
                df['Ticker'] = ticker
                all_results.append(df)
        except Exception as e:
            print(f"Error backtesting {ticker}: {e}")
            
    if not all_results:
        print("No trades found for any ticker.")
        return
        
    final_df = pd.concat(all_results, ignore_index=True)
    
    print("\n" + "="*50)
    print("FINAL SUMMARY PER TICKER")
    print("="*50)
    
    summary = []
    for ticker in ELITE_ORB_BUNDLE:
        t_df = final_df[final_df['Ticker'] == ticker]
        if t_df.empty:
            summary.append({'Ticker': ticker, 'Trades': 0, 'Win%': 0, 'PnL': 0})
            continue
            
        wins = len(t_df[t_df['PnL'] > 0])
        total = len(t_df)
        win_rate = (wins/total)*100
        pnl = t_df['PnL'].sum()
        
        summary.append({
            'Ticker': ticker,
            'Trades': total,
            'Win%': round(win_rate, 2),
            'PnL': round(pnl, 2)
        })
        
    sum_df = pd.DataFrame(summary)
    print(sum_df.to_string(index=False))
    
    print("\n" + "="*50)
    print("AGGREGATE RESULTS")
    print("="*50)
    total_trades = len(final_df)
    total_wins = len(final_df[final_df['PnL'] > 0])
    avg_win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = final_df['PnL'].sum()
    
    print(f"Total Portfolio Trades: {total_trades}")
    print(f"Overall Portfolio Win Rate: {avg_win_rate:.2f}%")
    print(f"Total Portfolio PnL (1 Qty per trade): {total_pnl:.2f}")
    
    final_df.to_csv("elite_bundle_backtest.csv", index=False)
    print("\nDetailed trade log saved to elite_bundle_backtest.csv")

if __name__ == "__main__":
    run_suite()
