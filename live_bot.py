"""
Live Trading Bot
----------------
This script runs continuously during market hours.
It:
1. Filters Nifty 50 stocks for 'Day Trading' Setups (CPR Trend + Narrow CPR)
2. Monitors them on 5-minute timeframe.
3. Logs 'Paper Trades' when signals occur.
"""

import time
import logging
import pandas as pd
from datetime import datetime
import os
from market_scanner import scan_stock, NIFTY_50
from day_trading_strategy import fetch_data, add_rsi, calculate_cpr
from ai_validator import AIValidator
from fyers_integration import FyersApp
from option_util import get_fyers_symbol

# Setup Logging
LOG_FILE = "bot.log"
TRADE_LOG = "trade_log.csv"
TRADE_COLUMNS = ['Time', 'Ticker', 'Signal', 'Entry_Price', 'Qty', 'SL', 'Target', 'Notes', 'Exit_Time', 'Exit_Price', 'Exit_Reason', 'PnL', 'Status']
CONFIG_FILE = "bot_config.json"

def load_config():
    """Load risk config from JSON."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except:
        # Defaults
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
LOG_FILE = "bot.log"
TRADE_LOG = "trade_log.csv"

# Risk Management Config
INITIAL_CAPITAL = 100000 # Total Account Capital (Reference)
FIXED_CAPITAL_PER_TRADE = 10000 # Max allocation per stock trade
MAX_TRADES_PER_DAY = 5
RISK_PER_TRADE = 0.01  # 1% (SL distance)
MAX_DAILY_LOSS = 0.02  # 2%

# Index Lot Sizes (Fixed 1 Lot)
NIFTY_LOT_SIZE = 65
BANKNIFTY_LOT_SIZE = 30

# Specific Index Limits
MAX_NIFTY_TRADES = 2
MAX_BANKNIFTY_TRADES = 2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def get_atm_strike(price, ticker):
    """Calculate ATM Strike Price."""
    if 'BANK' in ticker.upper():
        return int(round(price / 100) * 100)
    return int(round(price / 50) * 50)

def get_watchlist():
    """Run initial scan to find 'Stocks in Play'."""
    logging.info("Running Daily Pre-Market Scan...")
    watchlist = []
    
    watchlist = []
    
    # Always check Indices first
    indices = ['^NSEI', '^NSEBANK']
    for ticker in indices:
        try:
             res = scan_stock(ticker)
             if res: # Indices don't need Narrow CPR strictly, just Trend
                 watchlist.append({'ticker': ticker, 'trend': res['Trend']})
             else:
                 # Force add if scan fails, assuming NEUTRAL/Wait logic will handle it in main loop
                 # Better to have it and fail check_signals than miss it entirely
                 logging.warning(f"Index {ticker} initial scan failed, adding with default Neutral bias.")
                 watchlist.append({'ticker': ticker, 'trend': 'NEUTRAL'})
                 
        except Exception as e:
             logging.error(f"Error scanning index {ticker}: {e}. Forcing add.")
             watchlist.append({'ticker': ticker, 'trend': 'NEUTRAL'})

    # Simple scan (not threaded here to avoid complexity in bot loop)
    for ticker in NIFTY_50:
        try:
            # We use the scan_stock logic but just need Trend/Narrow status
            # scan_stock returns dict if successful
            res = scan_stock(ticker)
            if res and res['Narrow_CPR']:
                # ONLY Trade Narrow CPR days for high probability
                watchlist.append({
                    'ticker': ticker,
                    'trend': res['Trend'] # BULLISH or BEARISH
                })
        except Exception as e:
            logging.error(f"Error scanning stock {ticker}: {e}")
            continue
            
    logging.info(f"Watchlist created: {len(watchlist)} stocks found.")
    return watchlist

def check_for_signals(watchlist):
    """Check each stock in watchlist for entry signals."""
    current_time = datetime.now()
    if current_time.second < 10: # Log only once per minute
        # Explicitly mention indices to reassure user
        indices_in_watch = [t['ticker'] for t in watchlist if t['ticker'].startswith('^')]
        logging.info(f"Checking signals for {len(watchlist)} stocks including {indices_in_watch}...")

    MAX_STOCK_TRADES = 5
    
    # Load or Create Log
    if not os.path.exists(TRADE_LOG):
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(TRADE_LOG, index=False)
    
    try:
        df_log = pd.read_csv(TRADE_LOG)
        # Ensure all columns exist (start of day migration)
        for col in TRADE_COLUMNS:
            if col not in df_log.columns: df_log[col] = None
    except:
        df_log = pd.DataFrame(columns=TRADE_COLUMNS)

    # 1. Check Open Trades for Exits
    # Filter 'OPEN' status
    if not df_log.empty and 'Status' in df_log.columns:
        open_trades = df_log[df_log['Status'] == 'OPEN']
        
        # We can only check exits for stocks we have data for.
        # Luckily we iterate watchlist below. 
        # But we must check Exits BEFORE New Entries for the same stock (to free up cool-down?)
        # Actually cool-down prevents re-entry, but we want to close first.
        
        # Optimization: We loop watchlist anyway. Let's handle entries/exits per stock.
        pass
    
    today_trades = 0
    nifty_trades = 0
    banknifty_trades = 0
    
    if os.path.exists(TRADE_LOG):
        try:
            df_log = pd.read_csv(TRADE_LOG)
            if not df_log.empty and 'Time' in df_log.columns:
                df_log['Time'] = pd.to_datetime(df_log['Time'])
                today_trades = df_log[df_log['Time'].dt.date == datetime.now().date()]
                trades_today = len(today_trades)
                
                # Count Specifics
                nifty_trades = len(today_trades[today_trades['Ticker'] == '^NSEI'])
                banknifty_trades = len(today_trades[today_trades['Ticker'] == '^NSEBANK'])
        except Exception:
            pass
    
    # Load Dynamic Config
    config = load_config()
    MAX_STOCK_TRADES = config.get('MAX_STOCK_TRADES', 5)
    MAX_NIFTY_TRADES = config.get('MAX_NIFTY_TRADES', 2)
    MAX_BANKNIFTY_TRADES = config.get('MAX_BANKNIFTY_TRADES', 2)
    ALLOCATION = config.get('ALLOCATION_PER_TRADE', 10000)
    RISK_PCT = config.get('RISK_PER_TRADE_PERCENT', 1.0) / 100.0

    # Calculate Stock Trades
    stock_trades = trades_today - nifty_trades - banknifty_trades
    
    # Global cap removed in favor of segmented caps
    # but we can log status
    if current_time.second < 10:
        logging.info(f"Stats: Stocks {stock_trades}/{MAX_STOCK_TRADES}, Nifty {nifty_trades}/{MAX_NIFTY_TRADES}, BankNifty {banknifty_trades}/{MAX_BANKNIFTY_TRADES}")

    for item in watchlist:
        ticker = item['ticker']
        
        # Check Limits based on category
        if ticker == '^NSEI':
            if nifty_trades >= MAX_NIFTY_TRADES: continue
        elif ticker == '^NSEBANK':
            if banknifty_trades >= MAX_BANKNIFTY_TRADES: continue
        else:
            # It's a stock
            if stock_trades >= MAX_STOCK_TRADES: continue
        last_trade_time = None
        if os.path.exists(TRADE_LOG):
             try:
                 df_log = pd.read_csv(TRADE_LOG)
                 if not df_log.empty and 'Time' in df_log.columns:
                     df_log['Time'] = pd.to_datetime(df_log['Time'])
                     # Filter for this ticker today
                     today_trades = df_log[df_log['Time'].dt.date == datetime.now().date()]
                     ticker_trades = today_trades[today_trades['Ticker'] == ticker]
                     if not ticker_trades.empty:
                         last_trade_time = ticker_trades.iloc[-1]['Time']
             except Exception:
                 pass
        
        if last_trade_time:
            time_diff = (datetime.now() - last_trade_time).total_seconds() / 60
            if time_diff < 30: # 30 Minute Cool-down
                continue
        
        # Specific Index Limits Check
        if ticker == '^NSEI' and nifty_trades >= MAX_NIFTY_TRADES:
            continue # Skip Nifty if limit reached
        if ticker == '^NSEBANK' and banknifty_trades >= MAX_BANKNIFTY_TRADES:
            continue # Skip BankNifty if limit reached
        bias = item['trend']
        
        try:
            # Fetch Data (fast fetch)
            # We need Daily for CPR (already have bias but need levels)
            # We need Intraday
            daily_df, intraday_df = fetch_data(ticker)
            
            if intraday_df.empty or len(intraday_df) < 20: continue
            
            # Indicators
            daily_cpr = calculate_cpr(daily_df)
            
            # DYNAMIC BIAS CHECK: If Neutral (e.g. scan failed), recalc from daily data
            if bias == "NEUTRAL":
                try:
                    curr_cpr = daily_cpr.iloc[-1]
                    prev_cpr = daily_cpr.iloc[-2]
                    if curr_cpr['Pivot'] > prev_cpr['Pivot']:
                        bias = "BULLISH"
                    elif curr_cpr['Pivot'] < prev_cpr['Pivot']:
                        bias = "BEARISH"
                    logging.info(f"Updated Bias for {ticker}: {bias}")
                except:
                    pass
            
            intraday_df = add_rsi(intraday_df)
            intraday_df['Vol_SMA20'] = intraday_df['Volume'].rolling(20).mean()
            intraday_df['EMA20'] = intraday_df['Close'].ewm(span=20, adjust=False).mean()
            
            # ATR for Stop Loss
            high_low = intraday_df['High'] - intraday_df['Low']
            high_close = (intraday_df['High'] - intraday_df['Close'].shift()).abs()
            low_close = (intraday_df['Low'] - intraday_df['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            intraday_df['ATR'] = tr.rolling(14).mean()
            
            # VWAP
            intraday_df['Cum_Vol'] = intraday_df.groupby(intraday_df.index.date)['Volume'].cumsum()
            intraday_df['Cum_Vol_Price'] = intraday_df.groupby(intraday_df.index.date).apply(lambda x: (x['Close'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
            intraday_df['VWAP'] = intraday_df['Cum_Vol_Price'] / intraday_df['Cum_Vol']
            
            # --- EXIT LOGIC ---
            # Check if this stock has an OPEN trade
            current_price = intraday_df.iloc[-1]['Close']
            
            if not df_log.empty:
                # Find open trades for this ticker
                open_pos = df_log[(df_log['Ticker'] == ticker) & (df_log['Status'] == 'OPEN')]
                
                for idx, trade in open_pos.iterrows():
                    entry_price = trade['Entry_Price']
                    sl = trade['SL']
                    target_val = trade['Target'] # Could be string "TRAIL..."
                    signal_type = trade['Signal']
                    qty = trade['Qty']
                    
                    exit_reason = None
                    
                    # 1. Stop Loss Check
                    if "LONG" in signal_type and current_price <= sl:
                        exit_reason = "SL HIT"
                    elif "SHORT" in signal_type and current_price >= sl:
                        exit_reason = "SL HIT"
                        
                    # 2. Target Check (Fixed)
                    try:
                        tgt_price = float(target_val)
                        if "LONG" in signal_type and current_price >= tgt_price:
                            exit_reason = "TARGET HIT"
                        elif "SHORT" in signal_type and current_price <= tgt_price:
                            exit_reason = "TARGET HIT"
                    except:
                        # Trailing SL Logic (Simplified: Use EMA20 Close)
                        ema20 = intraday_df.iloc[-1]['EMA20']
                        if "TRAIL" in str(target_val):
                            if "LONG" in signal_type and current_price < ema20:
                                exit_reason = "TRAIL SL HIT"
                            elif "SHORT" in signal_type and current_price > ema20:
                                exit_reason = "TRAIL SL HIT"
                    
                    if exit_reason:
                        # Calculate PnL
                        if "LONG" in signal_type:
                            pnl = (current_price - entry_price) * qty
                        else:
                            pnl = (entry_price - current_price) * qty
                            
                        # Update Log
                        close_trade(idx, current_price, exit_reason, pnl)
                        logging.info(f"🚫 EXIT: {ticker} | Reason: {exit_reason} | PnL: {pnl:.2f}")

            # Map CPR
            
            # Map CPR
            daily_cpr['Date_Only'] = daily_cpr.index.date
            intraday_df['Date_Only'] = intraday_df.index.date
            
            # Get latest candle and CPR
            curr = intraday_df.iloc[-1]
            prev = intraday_df.iloc[-2]
            
            # Find matching CPR
            today_cpr = daily_cpr[daily_cpr['Date_Only'] == curr['Date_Only']]
            if today_cpr.empty: continue
            today_cpr = today_cpr.iloc[-1]
            
            tc = today_cpr['TC']
            bc = today_cpr['BC']
            upper_cpr = max(tc, bc)
            lower_cpr = min(tc, bc)
            
            # Logic
            signal = None
            price = curr['Close']
            stop_loss = 0.0
            target = 0.0
            atr = curr['ATR']
            
            
            if bias == "BULLISH":
                # Long Setup: Price > VWAP and (Breakout TC or RSI Cross)
                if price > curr['VWAP']:
                    breakout = prev['Close'] < upper_cpr and curr['Close'] > upper_cpr
                    rsi_cross = prev['RSI'] <= 60 and curr['RSI'] > 60
                    
                    if breakout or rsi_cross:
                        if curr['Volume'] > curr['Vol_SMA20']:
                            signal = "LONG"
                            stop_loss = price - (atr * 1.5)
                            target = price + (atr * 3) # 1:2 RR
                            
            elif bias == "BEARISH":
                # Short Setup: Price < VWAP
                if price < curr['VWAP']:
                    breakdown = prev['Close'] > lower_cpr and curr['Close'] < lower_cpr
                    rsi_cross = prev['RSI'] >= 40 and curr['RSI'] < 40
                    
                    if breakdown or rsi_cross:
                        if curr['Volume'] > curr['Vol_SMA20']:
                            signal = "SHORT"
                            stop_loss = price + (atr * 1.5)
                            target = price - (atr * 3) # 1:2 RR
                            
            if signal:
                # Prepare Technical Data for AI
                technicals = {
                    "Bias": bias,
                    "CPR": "Narrow" if item.get('trend') else "Normal",
                    "Close": price,
                    "VWAP": curr['VWAP'],
                    "RSI": curr['RSI'],
                    "Volume": curr['Volume'],
                    "Vol_SMA20": curr['Vol_SMA20'],
                    "Prev_Close": prev['Close'],
                    "Upper_CPR": upper_cpr,
                    "Lower_CPR": lower_cpr
                }
                
                # Verify with AI
                validator = AIValidator() # Reads env var
                ai_decision = validator.validate_trade(ticker, signal, price, technicals)
                
                if ai_decision['valid']:
                    # Get Strategy Mode
                    mode = ai_decision.get('mode', 'SNIPER')
                    
                    if mode == 'SURFER':
                        target_desc = "TRAIL EMA20"
                    else:
                        target_desc = f"{target:.2f}"


                    # Calculate Quantity
                    risk_amt = ALLOCATION * RISK_PCT
                    risk_per_share = abs(price - stop_loss)
                    qty = 0
                    
                    if risk_per_share > 0:
                        # Risk Based Sizing: qty = risk_amt / risk per share ? 
                        # OR Allocation Based: qty = ALLOCATION / price
                        
                        # User asked for "Allocation", so usually Fixed Amt.
                        # Let's check risk:
                        # If risk_per_share * (ALLOCATION/price) > risk_amt ?
                        
                        # Standard Practice: Fixed Allocation approach usually implies Max Investment.
                        if price > 0:
                            qty = int(ALLOCATION / price)
                        
                    if qty > 0:
                        if ticker.startswith('^'):
                            # Indices Options - FIXED 1 LOT
                            strike = get_atm_strike(price, ticker)
                            otype = "CE" if "LONG" in signal else "PE"
                            
                            # Fixed Lot System
                            if '^NSEBANK' in ticker:
                                qty = BANKNIFTY_LOT_SIZE
                            else:
                                qty = NIFTY_LOT_SIZE
                            
                            signal = f"{signal} [{strike} {otype}]"

                        else:
                            # Stock Equity - Fixed Capital
                            # Already calculated above based on ALLOCATION
                            pass

                        log_trade(ticker, signal, price, qty, stop_loss, target_desc, datetime.now(), f"[{mode}] {ai_decision['reason']}")

                        # LIVETRADE EXECUTION
                        if LIVE_TRADE and broker:
                            execute_order(ticker, signal, qty, stop_loss)

                else:
                    logging.info(f"AI REJECTED {ticker} {signal}: {ai_decision['reason']}")
                
        except Exception as e:
            logging.error(f"Error checking {ticker}: {e}")
            continue

def monitor_positions():
    """
    Placeholder for Position Management (TSL).
    In a real broker integration, this would fetch open positions and update SL orders.
    For Paper Trading log, we can't easily track state across loop iterations without a DB.
    Adding a simple log message about TSL Plan.
    """
    pass

def close_trade(idx, exit_price, reason, pnl):
    """Update trade log with exit details."""
    try:
        df = pd.read_csv(TRADE_LOG)
        df.at[idx, 'Exit_Time'] = datetime.now()
        df.at[idx, 'Exit_Price'] = exit_price
        df.at[idx, 'Exit_Reason'] = reason
        df.at[idx, 'PnL'] = pnl
        df.at[idx, 'Status'] = 'CLOSED'
        df.to_csv(TRADE_LOG, index=False)
    except Exception as e:
        logging.error(f"Failed to close trade: {e}")

def log_trade(ticker, signal, price, qty, sl, target, time, notes=""):
    """Log trade to CSV and Logger."""
    target_str = f"{target:.2f}" if isinstance(target, (int, float)) else str(target)
    msg = f"SIGNAL: {signal} {qty} x {ticker} @ {price:.2f} | SL: {sl:.2f} TGT: {target_str} | AI: {notes}"
    logging.info(msg)
    
    new_row = {
        'Time': time, 
        'Ticker': ticker, 
        'Signal': signal, 
        'Entry_Price': price, 
        'Qty': qty, 
        'SL': sl, 
        'Target': target, 
        'Notes': notes,
        'Exit_Time': None,
        'Exit_Price': None,
        'Exit_Reason': None,
        'PnL': 0.0,
        'Status': 'OPEN'
    }
    
    df_new = pd.DataFrame([new_row])
    header = not os.path.exists(TRADE_LOG)
    df_new.to_csv(TRADE_LOG, mode='a', header=header, index=False)

def execute_order(ticker, signal, qty, stop_loss):
    """Execute Live Order on Fyers."""
    try:
        side = "BUY" if "LONG" in signal else "SELL"
        
        if ticker.startswith('^'):
            # Calculate Option Symbol
            # strike = get_atm_strike(curr['Close'], ticker) # Need current price, passed as arg? 
            # Note: execute_order doesn't have price. 
            # We must detect Strike from signal string or pass it?
            # Current Signal: "LONG [25000 CE]"
            
            try:
                import re
                # Extract Strike and Type from Signal String
                # Format: "LONG [25000 CE]"
                match = re.search(r'\[(\d+)\s+(CE|PE)\]', signal)
                if match:
                    strike = int(match.group(1))
                    otype = match.group(2)
                    
                    fyers_symbol = get_fyers_symbol(ticker, strike, otype)
                    
                    # Lot Size Logic
                    lot_size = 30 if 'BANK' in ticker else 65 # Updated Lot Sizes
                    # Adjust Qty to be multiple of Lot Size
                    qty = max(lot_size, int(round(qty / lot_size) * lot_size))
                    
                    logging.info(f"🎯 OPTION SYMBOL: {fyers_symbol} (Lot: {lot_size}, Qty: {qty})")
                    side = "BUY" # Options are always Bought for Long/Short logic here?
                    # Strategy Logic:
                    # Long Signal -> Buy CE (Side BUY)
                    # Short Signal -> Buy PE (Side BUY)
                    # Correct. We always BUY options.
                    side = "BUY" 
                    
                else:
                    logging.error(f"Could not parse Option details from: {signal}")
                    return
            except Exception as e:
                logging.error(f"Option Construct Create Failed: {e}")
                return
        
        else:
            # Equity Execution
            fyers_symbol = f"NSE:{ticker}-EQ"

        # 1. Place Market Entry
        logging.info(f"🚀 EXECUTING LIVE: {side} {qty} {fyers_symbol}")
        res = broker.place_order(fyers_symbol, int(qty), side, order_type="MARKET", product_type="INTRADAY")
        logging.info(f"Order Response: {res}")
        
        # 2. Place Stop Loss (Simplified SL-M)
        if 'id' in res: 
            # For Options, SL is critical.
            # SL-M supported.
            sl_side = "SELL" # Since we always BUY options
            
            # SL Calculation for Options is tricky without Premium Price.
            # We can't place SL-M immediately without knowing execution price.
            # Ideally, wait for trade update or use BO.
            # For MVP: Just Log warning.
            logging.warning("⚠️ Option SL not placed (Need Execution Price). Manage Manually!")
            
    except Exception as e:
        logging.error(f"Live Execution Failed: {e}")

def main():
    logging.info("--- Live Trading Bot Started ---")
    
    # 1. Build Watchlist
    watchlist = get_watchlist()
    if not watchlist:
        logging.warning("No stocks found matching criteria today. Exiting.")
        return

    logging.info("Starting Monitoring Loop (Press Ctrl+C to Stop)...")
    
    try:
        while True:
            # Check market hours (optional, omitted for testing capability)
            check_for_signals(watchlist)
            
            # Sleep 5 seconds for faster reaction
            time.sleep(5) 
            
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.error(f"Bot Crashed: {e}")

if __name__ == "__main__":
    main()
