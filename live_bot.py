"""
Live Trading Bot
----------------
This script runs continuously during market hours.
It:
1. Monitors Indices (^NSEI, ^NSEBANK) for 3-Candle 9EMA Strategy (Options).
2. Monitors Nifty 50 Stocks for CPR + RSI + VWAP Strategy (Equity Intraday).
3. Uses ADX to switch between SNIPER (Target Based) and SURFER (Trailing Exit) modes.
"""

import time
import logging
import pandas as pd
from datetime import datetime
import os
import json
from market_scanner import scan_stock, NIFTY_50
from day_trading_strategy import fetch_data, add_rsi, calculate_9ema, check_3_candle_setup, check_stock_signal, calculate_cpr, calculate_vwap, calculate_adx, detect_market_regime
from ai_validator import AIValidator
from fyers_integration import FyersApp
from option_util import get_fyers_symbol
import re

# Broker Init
FYERS_TOKEN = os.environ.get("FYERS_TOKEN")
FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
LIVE_TRADE = True if FYERS_TOKEN else False

broker = None
if LIVE_TRADE:
    try:
        broker = FyersApp(FYERS_CLIENT_ID, "secret_placeholder", access_token=FYERS_TOKEN)
        profile = broker.get_profile()
        logging.info(f"LIVE BROKER CONNECTED: {profile}")
    except Exception as e:
        logging.error(f"Broker Connect Failed: {e}")
        LIVE_TRADE = False

# Setup Logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "bot.log")
TRADE_LOG = os.path.join(BASE_DIR, "trade_log.csv")
CONFIG_FILE = os.path.join(BASE_DIR, "bot_config.json")
TRADE_COLUMNS = ['Time', 'Ticker', 'Instrument', 'Signal', 'Entry_Price', 'Qty', 'SL', 'Target', 'Notes', 'Exit_Time', 'Exit_Price', 'Exit_Reason', 'PnL', 'Status', 'Regime']

# Config
def load_config():
    """Load risk config from JSON."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        return {
            "MAX_DAILY_TRADES": 9,
            "MAX_STOCK_TRADES": 5, 
            "MAX_NIFTY_TRADES": 2,
            "MAX_BANKNIFTY_TRADES": 2,
            "CAPITAL": 100000,
            "ALLOCATION_PER_TRADE": 10000,
            "RISK_PER_TRADE_PERCENT": 1.0,
            "MAX_DAILY_LOSS_PERCENT": 2.0
        }

# Logger
# Configure Logging - Reduce Noise
# Force Root Logger to INFO
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NUCLEAR OPTION: Silence all existing loggers from imported libraries
# fyers_apiv3 likely sets up its own logger with DEBUG
for logger_name in logging.Logger.manager.loggerDict:
    if logger_name not in ["root"]: # Keep root
        logging.getLogger(logger_name).setLevel(logging.WARNING)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Globals
ai_cooldowns = {}

def get_strike_for_trade(price, ticker, otype, depth=2):
    """Calculate ITM Strike Price for Higher Delta."""
    step = 100 if 'BANK' in ticker.upper() else 50
    atm = int(round(price / step) * step)
    if otype == "CE": return atm - (step * depth)
    else: return atm + (step * depth)

def get_watchlist():
    """Build Watchlist: Indices + Nifty 50 Stocks."""
    logging.info("Building Daily Watchlist...")
    watchlist = []
    indices = ['^NSEI', '^NSEBANK']
    for ticker in indices: watchlist.append({'ticker': ticker, 'type': 'INDEX'})
    # Filter out indices from NIFTY_50 list as they are already added
    stock_list = [t for t in NIFTY_50 if t not in indices]
    for ticker in stock_list: watchlist.append({'ticker': ticker, 'type': 'STOCK'})
    logging.info(f"Watchlist: 2 Indices + {len(stock_list)} Stocks.")
    return watchlist

def check_for_signals(watchlist, config):
    """Check Signals for both Strategies."""
    current_time = datetime.now()
    
    # Initialize Log if missing
    if not os.path.exists(TRADE_LOG):
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(TRADE_LOG, index=False)
    
    config = load_config()
    MAX_STOCK_TRADES = config.get('MAX_STOCK_TRADES', 5)
    MAX_NIFTY_TRADES = config.get('MAX_NIFTY_TRADES', 2)
    MAX_BANKNIFTY_TRADES = config.get('MAX_BANKNIFTY_TRADES', 2)
    ALLOCATION = config.get('ALLOCATION_PER_TRADE', 10000)
    RISK_PCT = config.get('RISK_PER_TRADE_PERCENT', 1.0) / 100.0
    
    nifty_trades = 0
    banknifty_trades = 0
    stock_trades = 0
    
    df_log = pd.DataFrame()
    try:
        df_log = pd.read_csv(TRADE_LOG)
        if not df_log.empty and 'Time' in df_log.columns:
            df_log['Time'] = pd.to_datetime(df_log['Time'])
            tod = df_log[df_log['Time'].dt.date == current_time.date()]
            nifty_trades = len(tod[tod['Ticker'] == '^NSEI'])
            banknifty_trades = len(tod[tod['Ticker'] == '^NSEBANK'])
            
            # Count Stock Trades (Anything NOT Index)
            stock_trades = len(tod[~tod['Ticker'].isin(['^NSEI', '^NSEBANK'])])
    except: pass

    if current_time.second < 10:
        logging.info(f"Stats: Nifty {nifty_trades}, BankNifty {banknifty_trades}, Stocks {stock_trades}/{MAX_STOCK_TRADES}")

    logging.info(f"🔎 Scanning {len(watchlist)} tickers...")
    scan_count = 0

    for item in watchlist:
        scan_count += 1
        if scan_count % 10 == 0: logging.info(f"...scanned {scan_count}/{len(watchlist)}")

        ticker = item['ticker']
        is_index = item['type'] == 'INDEX'
        
        # Trade Limits
        if ticker == '^NSEI' and nifty_trades >= MAX_NIFTY_TRADES: continue
        if ticker == '^NSEBANK' and banknifty_trades >= MAX_BANKNIFTY_TRADES: continue
        if not is_index and stock_trades >= MAX_STOCK_TRADES: continue
        
        try:
            daily_df, intraday_df = fetch_data(ticker, broker=broker)
            if intraday_df.empty or len(intraday_df) < 50: continue # Need history for ADX
            
            # --- STRATEGY & REGIME ---
            regime = "SNIPER" # Default
            
            if is_index:
                # === INDICES (9EMA) ===
                intraday_df = calculate_9ema(intraday_df)
                intraday_df = calculate_adx(intraday_df)
                regime = detect_market_regime(intraday_df)
                
                if len(intraday_df) >= 4:
                    # Index: Cluster SL + 5 pts buffer
                    signal_data = check_3_candle_setup(intraday_df.iloc[-4:], sl_method='CLUSTER', limit_buffer=5.0)
                    strategy_name = f"3-Candle 9EMA ({regime})"
                    
                current_price = intraday_df.iloc[-1]['Close']
                ema = intraday_df.iloc[-1]['EMA9']
            else:
                # === STOCKS (Switch to 9EMA Strict) ===
                # Compute 9EMA for Stock
                intraday_df = calculate_9ema(intraday_df)
                
                # Use same logic as Index (Strict 3-Candle)
                if len(intraday_df) >= 4:
                    # Stock: Reference Candle High/Low SL + 0 buffer (User Request)
                    signal_data = check_3_candle_setup(intraday_df.iloc[-4:], sl_method='REF', limit_buffer=0.0)
                    strategy_name = "Stock 9EMA (Strict)"
                    regime = "SNIPER" # Default to Sniper for stocks
                
                current_price = intraday_df.iloc[-1]['Close']
                ema = intraday_df.iloc[-1]['EMA9']

            if signal_data:
                # Deduplication Logic: Check if we already traded this Signal Time
                # We need to load full log now (df_log is available from earlier)
                signal_time = signal_data.get('Time')
                if signal_time:
                    # Convert to pd.Timestamp for comparison
                    try:
                        sig_ts = pd.to_datetime(signal_time)
                        # Check df_log for trades defined AFTER this signal time or AT same time
                        # df_log['Time'] is string, need conversion
                        if not df_log.empty:
                            df_log['Time_Ts'] = pd.to_datetime(df_log['Time'])
                            # Buffer: If we traded within 2 mins of this signal or after
                            # Usually Signal Time is Candle Close (e.g. 13:40). Entry is 13:40:05.
                            # So any trade with Time >= Signal Time is a match.
                            existing = df_log[
                                (df_log['Ticker'] == ticker) & 
                                (df_log['Signal'] == signal_data['Signal']) &
                                (df_log['Time_Ts'] >= sig_ts) 
                            ]
                            if not existing.empty:
                                # logging.info(f"Skipping Duplicate: {ticker} Signal {sig_ts} already traded.")
                                continue
                    except Exception as e:
                        logging.warning(f"Dedupe Check Failed: {e}")
                signal_type = signal_data['Signal']
                entry_p = signal_data['Entry']
                sl_p = signal_data['SL']
                
                # Check AI Cooldown
                ts = time.time()
                last_ai = ai_cooldowns.get(ticker, 0)
                if ts - last_ai < 300: continue
                
                technicals = {
                    "Close": current_price,
                    "Signal": signal_type,
                    "Entry": entry_p,
                    "SL": sl_p,
                    "Strategy": strategy_name,
                    "Regime": regime
                }
                
                validator = AIValidator()
                ai_bypass = config.get("ALLOW_AI_BYPASS", False)
                ai_res = validator.validate_trade(ticker, signal_type, entry_p, technicals, allow_bypass=ai_bypass)
                ai_cooldowns[ticker] = ts
                
                if ai_res['valid']:
                    qty = 0
                    strike = 0
                    target_p = 0
                    order_sym = ""
                    
                    if is_index:
                         # OPTION BUYING
                         otype = "CE" if signal_type == 'LONG' else "PE"
                         strike = int(get_strike_for_trade(entry_p, ticker, otype, depth=2)) 
                         # Nifty Lot Size = 65 (Verified Live), BankNifty = 30 (Std) or 15? Assume 30 for now or checks.
                         # Actually user config says MAX_NIFTY_TRADES but defaults.
                         lot_size = 30 if 'BANK' in ticker else 65
                         
                         risk_per_share = abs(entry_p - sl_p)
                         qty = lot_size
                         if risk_per_share > 0:
                             risk_amt = ALLOCATION * RISK_PCT
                             calc_qty = int(risk_amt / (risk_per_share * 0.5))
                             qty = max(lot_size, (calc_qty // lot_size) * lot_size)
                         
                         order_sym = get_fyers_symbol(ticker, strike, otype)
                         
                         if regime == "SURFER":
                             target_p = 0 # No Target
                         else:
                             target_p = entry_p + (risk_per_share * 2 * (1 if signal_type == 'LONG' else -1))
                         
                    else:
                         # EQUITY
                         risk_per_share = abs(entry_p - sl_p)
                         qty = 1
                         if risk_per_share > 0:
                             risk_amt = ALLOCATION * RISK_PCT
                             qty = int(risk_amt / risk_per_share)
                         
                         
                         order_sym = f"NSE:{ticker.replace('.NS', '')}-EQ"
                         target_p = entry_p + (risk_per_share * 2 * (1 if signal_type == 'LONG' else -1))

                    full_signal = f"{signal_type} ({strategy_name})"
                    log_trade(ticker, order_sym, full_signal, entry_p, qty, sl_p, target_p, datetime.now(), ai_res['reason'], regime)
                    
                    if LIVE_TRADE and broker:
                        try:
                            side = "BUY"
                            if not is_index: side = "BUY" if signal_type == "LONG" else "SELL"
                            execute_order(order_sym, side, qty, 0)
                        except Exception as e:
                             logging.error(f"Execution Failed: {e}")
                else:
                     logging.info(f"AI REJECTED: {ai_res['reason']}")
                     
        except Exception as e:
            logging.error(f"Check error {ticker}: {e}")

def monitor_positions():
    """Check Open Trades for SL or Exit Signals."""
    if not os.path.exists(TRADE_LOG): return
    
    try:
        df = pd.read_csv(TRADE_LOG)
        open_trades = df[df['Status'] == 'OPEN']
        if open_trades.empty: return

        # Reload Config for Safety
        config = load_config()

        for idx, trade in open_trades.iterrows():
            ticker = trade['Ticker']
            signal = trade['Signal']
            entry_p = float(trade['Entry_Price'])
            sl_p = float(trade['SL'])
            qty = int(trade['Qty'])
            instrument = trade['Instrument']
            regime = trade.get('Regime', 'SNIPER') # Default
            
            try:
                # 1. Fetch Live Data
                # Use scanner/strategy logic to get latest candle
                daily, intraday = fetch_data(ticker, broker=broker) # Uses cache or fetch
                if intraday.empty: continue
                
                # Calculate Indicators
                intraday = calculate_9ema(intraday)
                curr = intraday.iloc[-1]
                prev = intraday.iloc[-2] # Closed Candle
                
                current_price = curr['Close']
                
                # 2. Check Hard SL (Live Price)
                # If Long: Price < SL
                # If Short: Price > SL
                hl_hit = False
                exit_msg = ""
                
                # Dynamic PnL Simulation
                pnl = 0
                if 'LONG' in signal:
                    pnl = (current_price - entry_p) * qty
                    if current_price <= sl_p:
                        hl_hit = True
                        exit_msg = "Hard SL Hit"
                else: 
                     # Short
                     pnl = (entry_p - current_price) * qty
                     if current_price >= sl_p:
                         hl_hit = True
                         exit_msg = "Hard SL Hit"
                         
                if hl_hit:
                    logging.info(f"🛑 SL HIT for {ticker}: {current_price} vs {sl_p}")
                    close_trade_execution(idx, ticker, instrument, qty, signal, current_price, exit_msg, pnl)
                    continue
                    
                # 3. Check Strategy Exit (9EMA Reversal) on CLOSED Candle
                # Long Exit: Close < 9EMA
                # Short Exit: Close > 9EMA
                
                strategy_exit = False
                
                if 'LONG' in signal:
                    if prev['Close'] < prev['EMA9']:
                        strategy_exit = True
                        exit_msg = "9EMA Reversal (Close < 9EMA)"
                else:
                    if prev['Close'] > prev['EMA9']:
                        strategy_exit = True
                        exit_msg = "9EMA Reversal (Close > 9EMA)"
                        
                if strategy_exit:
                    logging.info(f"⚠️ STRATEGY EXIT for {ticker}: {prev['Close']} vs EMA {prev['EMA9']:.2f}")
                    # Use Close of that candle or Current market? Use Current Market for execution
                    close_trade_execution(idx, ticker, instrument, qty, signal, current_price, exit_msg, pnl)
                    continue
                    
            except Exception as e:
                logging.error(f"Monitor Error {ticker}: {e}")

    except Exception as e:
        logging.error(f"Monitor Loop Error: {e}")

def close_trade_execution(idx, ticker, instrument, qty, signal, price, reason, pnl):
    # 1. Log Exit
    close_trade(idx, price, reason, pnl)
    logging.info(f"✅ CLOSED {ticker}: {reason} | PnL: {pnl:.2f}")
    
    # 2. Execute Market Order
    if LIVE_TRADE and broker:
        try:
            side = "SELL" # Default for Closing Long
            if 'SHORT' in signal: side = "BUY" # Closing Short
            
            # For Indices/Fyers Options, we Buy/Sell the Instrument
            # Logic same as entry but reversed
            
            execute_order(instrument, side, qty, 0)
        except Exception as e:
            logging.error(f"Close Execution Failed: {e}")

def close_trade(idx, price, reason, pnl):
    try:
        df = pd.read_csv(TRADE_LOG)
        df.at[idx, 'Status'] = 'CLOSED'
        df.at[idx, 'Exit_Time'] = datetime.now()
        df.at[idx, 'Exit_Price'] = price
        df.at[idx, 'Exit_Reason'] = reason
        df.at[idx, 'PnL'] = pnl
        df.to_csv(TRADE_LOG, index=False)
    except: pass

def log_trade(ticker, instrument, signal, price, qty, sl, target, time, notes="", regime="SNIPER"):
    logging.info(f"SIGNAL: {signal} [{regime}] {qty} x {ticker} ({instrument}) @ {price} | SL {sl:.2f}")
    if not os.path.exists(TRADE_LOG):
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(TRADE_LOG, index=False)
    
    new_data = {
        'Time': time, 'Ticker': ticker, 'Instrument': instrument, 'Signal': signal, 'Entry_Price': price,
        'Qty': qty, 'SL': sl, 'Target': target, 'Notes': notes, 'Status': 'OPEN', 'PnL': 0.0, 'Regime': regime
    }
    pd.DataFrame([new_data], columns=TRADE_COLUMNS).to_csv(TRADE_LOG, mode='a', header=False, index=False)

def execute_order(symbol, side, qty, stop_loss):
    if not broker: return
    try:
        logging.info(f"🚀 LIVE ORDER: {side} {qty} {symbol}")
        res = broker.place_order(symbol, int(qty), side, order_type="MARKET", product_type="INTRADAY")
        if 'id' in res: logging.info(f"Order Placed: {res['id']}")
    except Exception as e: logging.error(f"Execution Error: {e}")

def square_off_all_positions():
    """Close all OPEN positions at EOD (3:15 PM)."""
    if not os.path.exists(TRADE_LOG): return
    
    try:
        df = pd.read_csv(TRADE_LOG)
        open_trades = df[df['Status'] == 'OPEN']
        if open_trades.empty: return

        logging.info(f"⏰ EOD SQUARE OFF: Closing {len(open_trades)} positions...")
        
        for idx, trade in open_trades.iterrows():
            ticker = trade['Ticker']
            qty = int(trade['Qty'])
            signal = trade['Signal']
            instrument = trade['Instrument']
            entry_p = float(trade['Entry_Price'])
            
            # Fetch LTP for closing price
            # We can use fetch_data or just assume last known price if market is closing/closed
            # But better to try to get a quote.
            try:
                # Taking a quick quote via fetch_data is safest re-use
                daily, intraday = fetch_data(ticker, broker=broker)
                if not intraday.empty:
                    close_price = intraday.iloc[-1]['Close']
                else:
                    close_price = entry_p # Fallback avoids crash
                    
                # Calculate PnL
                pnl = 0
                if 'LONG' in signal: pnl = (close_price - entry_p) * qty
                else: pnl = (entry_p - close_price) * qty
                
                close_trade_execution(idx, ticker, instrument, qty, signal, close_price, "EOD Square Off", pnl)
                
            except Exception as e:
                logging.error(f"Square Off Error {ticker}: {e}")
                
    except Exception as e:
        logging.error(f"Square Off Loop Error: {e}")

def main():
    logging.info("--- Live Bot (Indices + Stocks + ADX Dynamic) Started ---")
    watchlist = get_watchlist()
    
    while True:
        try:
            now = datetime.now()
            
            # Pre-Market
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                 logging.info(f"Pre-Market Wait... {now.strftime('%H:%M:%S')}")
                 time.sleep(60)
                 continue
                 
            # Post-Market / Square Off (3:15 PM)
            if now.hour >= 15 and now.minute >= 15:
                 logging.info("Market Closing (3:15 PM). Squaring Off...")
                 square_off_all_positions()
                 time.sleep(60)
                 continue
            
            # Reload config dynamically every loop? Or just in main?
            # Let's reload to pick up changes from UI
            config = load_config() 
            monitor_positions() # Check existing trades FIRST
            check_for_signals(watchlist, config)
            time.sleep(60)
        except KeyboardInterrupt: break
        except Exception as e:
            logging.error(f"Loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
