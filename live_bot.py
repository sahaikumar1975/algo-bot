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
INITIAL_CAPITAL = 100000
RISK_PER_TRADE = 0.01  # 1%
MAX_DAILY_LOSS = 0.02  # 2%

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
        except Exception as e:
             logging.error(f"Error scanning index {ticker}: {e}")
             continue

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
        logging.info(f"Checking signals for {len(watchlist)} stocks...")

    for item in watchlist:
        ticker = item['ticker']
        bias = item['trend']
        
        try:
            # Fetch Data (fast fetch)
            # We need Daily for CPR (already have bias but need levels)
            # We need Intraday
            daily_df, intraday_df = fetch_data(ticker)
            
            if intraday_df.empty or len(intraday_df) < 20: continue
            
            # Indicators
            daily_cpr = calculate_cpr(daily_df)
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
                    risk_amt = INITIAL_CAPITAL * RISK_PER_TRADE
                    risk_per_share = abs(price - stop_loss)
                    qty = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
                    
                    if qty > 0:
                        if ticker.startswith('^'):
                            # Indices Options
                            strike = get_atm_strike(price, ticker)
                            # Symbol Format: NSE:NIFTY23OCT19500CE (Example - needs logic)
                            # Simplified for now -> Just Log strike
                            otype = "CE" if "LONG" in signal else "PE"
                            # Fyers Symbol Construct (Needs Expiry - Complex)
                            # For MVP: We will Trade FUTURE or just LOG
                            signal = f"{signal} [{strike} {otype}]"

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

def log_trade(ticker, signal, price, qty, sl, target, time, notes=""):
    """Log trade to CSV and Logger."""
    target_str = f"{target:.2f}" if isinstance(target, (int, float)) else str(target)
    msg = f"SIGNAL: {signal} {qty} x {ticker} @ {price:.2f} | SL: {sl:.2f} TGT: {target_str} | AI: {notes}"
    logging.info(msg)
    
    # Check if we already logged this recently (debounce) to avoid spam
    # Implementation optional, for now just append
    
    df_new = pd.DataFrame([{
        'Time': time, 'Ticker': ticker, 'Signal': signal, 
        'Price': price, 'Qty': qty, 'SL': sl, 'Target': target, 'Notes': notes
    }])
    
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
            
            # Sleep 1 minute
            time.sleep(60) 
            
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.error(f"Bot Crashed: {e}")

if __name__ == "__main__":
    main()
