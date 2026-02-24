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
from dotenv import load_dotenv
load_dotenv(override=True)

from market_scanner import scan_stock, NIFTY_50
from day_trading_strategy import (
    fetch_data, add_rsi, calculate_ema, check_3_candle_setup, check_orb_breakout_setup,
    check_stock_signal, calculate_cpr, calculate_vwap, calculate_adx, detect_market_regime,
    train_hmm_model, get_hmm_regime
)
from ai_validator import AIValidator
from fyers_integration import FyersApp
from option_util import get_fyers_symbol, get_futures_symbol
from strategies.pos_futures import PositionalFuturesBot
import re

# Broker Init
FYERS_TOKEN = os.environ.get("FYERS_TOKEN")
FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
LIVE_TRADE = True if FYERS_TOKEN else False

broker = None
if LIVE_TRADE:
    try:
        broker = FyersApp(FYERS_CLIENT_ID, "secret_placeholder", access_token=FYERS_TOKEN)
        # Verify Token via API Call
        profile = broker.get_profile()
        
        if 'error' in profile or profile.get('s') == 'error':
             logging.error(f"❌ LIVE CONN FAILED: {profile}")
             print(f"❌ LIVE CONN FAILED: {profile}")
             LIVE_TRADE = False
        else:
             logging.info(f"✅ LIVE BROKER CONNECTED: {profile.get('data', {}).get('name', 'User')}")
    except Exception as e:
        logging.error(f"❌ Broker Connect Exception: {e}")
        LIVE_TRADE = False

if not LIVE_TRADE and FYERS_TOKEN:
    logging.warning("⚠️ FALLING BACK TO PAPER MODE due to Auth Failure.")
    print("⚠️ FALLING BACK TO PAPER MODE due to Auth Failure.")

# Setup Logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "bot.log")
TRADE_LOG = os.path.join(BASE_DIR, "trade_log.csv")
CONFIG_FILE = os.path.join(BASE_DIR, "bot_config.json")
TRADE_COLUMNS = ['Time', 'Ticker', 'Instrument', 'Signal', 'Entry_Price', 'Qty', 'SL', 'Target', 'Notes', 'Exit_Time', 'Exit_Price', 'Exit_Reason', 'PnL', 'Status', 'Regime', 'Charges', 'Net_PnL', 'Actual_Entry', 'Actual_Exit']

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
            "MAX_TRADES_PER_STOCK": 2,
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
file_handler = logging.FileHandler(LOG_FILE, mode='a')
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
    """Calculate ITM/ATM Strike Price for Options."""
    step = 50
    t = ticker.upper()
    if 'BANK' in t: step = 100
    elif 'BHARTIARTL' in t: step = 10
    elif 'DLF' in t: step = 5
    elif 'TVSMOTOR' in t: step = 10
    elif 'LT' in t: step = 20 # LT strikes are usually 20 or 50
    elif 'HINDUNILVR' in t: step = 20
    
    atm = int(round(price / step) * step)
    if otype == "CE": return atm - (step * depth)
    else: return atm + (step * depth)

def calculate_charges(entry_price, exit_price, qty, ticker, is_option):
    """Calculate Approx Charges (Brokerage, STT, Txn, GST, SEBI, Stamp)."""
    # Turnover
    entry_turnover = entry_price * qty
    exit_turnover = exit_price * qty
    total_turnover = entry_turnover + exit_turnover
    
    # 1. Brokerage (Fyers: Max 20 per order)
    brokerage_entry = 20.0
    brokerage_exit = 20.0
    
    if not is_option:
        # Equity Intraday: Min(20, 0.03%)
        brokerage_entry = min(20.0, entry_turnover * 0.0003)
        brokerage_exit = min(20.0, exit_turnover * 0.0003)
    else:
        # OPTION CHECK: If Price is Spot Level, normalize to estimated premium.
        # ATM options are roughly 0.6%-0.8% of index value or 1.5%-2.5% of stock value (monthly)
        t = ticker.upper()
        if 'BANK' in t and entry_price > 5000:
             entry_price = entry_price * 0.008
             exit_price = exit_price * 0.008
        elif 'NIFTY' in t and entry_price > 5000:
             entry_price = entry_price * 0.006
             exit_price = exit_price * 0.006
        elif entry_price > 100:
             entry_price = entry_price * 0.02
             exit_price = exit_price * 0.02
        # Recalculate Turnover
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
    txn_rate = 0.0005 if is_option else 0.0000325
    txn_charges = total_turnover * txn_rate
    
    # 4. GST (18% on Brokerage + Txn Charges)
    gst = (total_brokerage + txn_charges) * 0.18
    
    # 5. SEBI Charges (10 per crore = 0.0001%)
    sebi_charges = total_turnover * 0.000001
    
    # 6. Stamp Duty (0.003% on Buy only)
    stamp_duty = entry_turnover * 0.00003
    
    total_charges = total_brokerage + stt + txn_charges + gst + sebi_charges + stamp_duty
    return round(total_charges, 2)

def check_and_migrate_log():
    try:
        if os.path.exists(TRADE_LOG):
            df = pd.read_csv(TRADE_LOG)
            migrated = False
            for col in TRADE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0 if col in ['Charges', 'Net_PnL', 'Actual_Entry', 'Actual_Exit'] else ""
                    migrated = True
            
            if migrated:
                logging.info("Migrating Trade Log schema...")
                df.to_csv(TRADE_LOG, index=False)
    except: pass

def get_watchlist():
    """Build Watchlist: Indices + Nifty 50 Stocks based on Config."""
    config = load_config()
    enable_indices = config.get('ENABLE_INDICES', True)
    enable_stocks = config.get('ENABLE_STOCKS', False)
    enable_elite_orb_bundle = config.get('ENABLE_ELITE_ORB_BUNDLE', True)
    
    logging.info(f"Building Watchlist... [Indices: {enable_indices}, Nifty50: {enable_stocks}, ELITE-ORB: {enable_elite_orb_bundle}]")
    watchlist = []
    
    indices = ['^NSEI', '^NSEBANK']
    if enable_indices:
        for ticker in indices: watchlist.append({'ticker': ticker, 'type': 'INDEX'})
        
    stock_tickers = set()
    if enable_stocks:
        stock_tickers.update([t for t in NIFTY_50 if t not in indices])
        
    if enable_elite_orb_bundle:
        # Nifty 100 Filtered Elite Bundle (Top 10 High PnL + >50% WR)
        ELITE_ORB_BUNDLE = [
            # Original Nifty 100 Elite
            "MARUTI.NS", "APOLLOHOSP.NS", "LTIM.NS", "SIEMENS.NS", 
            "ASIANPAINT.NS", "TVSMOTOR.NS", "BHARTIARTL.NS", 
            "HINDUNILVR.NS", "TCS.NS", "DLF.NS",
            # New Nifty 200 Top Performers (Phase 6 Screen)
            "GODREJPROP.NS", "LT.NS", "HINDALCO.NS", "GODREJCP.NS"
        ]
        stock_tickers.update(ELITE_ORB_BUNDLE)
        
    if stock_tickers:
        for ticker in stock_tickers: watchlist.append({'ticker': ticker, 'type': 'STOCK'})
        
    logging.info(f"Watchlist: {len([x for x in watchlist if x['type']=='INDEX'])} Indices + {len([x for x in watchlist if x['type']=='STOCK'])} Stocks.")
    return watchlist

def check_for_signals(watchlist, config):
    """Check Signals for both Strategies."""
    current_time = datetime.now()
    
    # Initialize Log if missing or migrate columns
    check_and_migrate_log()
    
    config = load_config()
    MAX_STOCK_TRADES = config.get('MAX_STOCK_TRADES', 5)
    MAX_TRADES_PER_STOCK = config.get('MAX_TRADES_PER_STOCK', 2)
    MAX_NIFTY_TRADES = config.get('MAX_NIFTY_TRADES', 2)
    MAX_BANKNIFTY_TRADES = config.get('MAX_BANKNIFTY_TRADES', 2)
    ALLOCATION_STOCK = config.get('ALLOCATION_STOCK', 10000)
    ALLOCATION_INDEX = config.get('ALLOCATION_INDEX', 33000)
    RISK_PCT = config.get('RISK_PER_TRADE_PERCENT', 1.0) / 100.0
    
    nifty_trades = 0
    banknifty_trades = 0
    stock_trades = 0
    
    df_log = pd.DataFrame()

    tod = pd.DataFrame()
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
        ALLOCATION = ALLOCATION_INDEX if is_index else ALLOCATION_STOCK
        
        # Trade Limits
        if ticker == '^NSEI' and nifty_trades >= MAX_NIFTY_TRADES: continue
        if ticker == '^NSEBANK' and banknifty_trades >= MAX_BANKNIFTY_TRADES: continue
        if not is_index and stock_trades >= MAX_STOCK_TRADES: continue
        
        # Per Stock Limit
        if not is_index:
             curr_stock_trades = 0
             if not tod.empty:
                 curr_stock_trades = len(tod[tod['Ticker'] == ticker])
             if curr_stock_trades >= MAX_TRADES_PER_STOCK: continue
        
        interval_fyers = "5" if is_index else "15"
        interval_yf = "5m" if is_index else "15m"
        
        try:
            daily_df, intraday_df = fetch_data(ticker, broker=broker, interval_fyers=interval_fyers, interval_yf=interval_yf)
            if intraday_df.empty or len(intraday_df) < 50: continue # Need history for ADX
            
            # --- SIGNAL CONFIRMATION: Exclude Hybrid Rows ---
            # If the last candle's timestamp is not perfectly aligned with the interval,
            # it's a "Hybrid" live quote. We only use CLOSED candles for signal triggers.
            signal_df = intraday_df.copy()
            from day_trading_strategy import calculate_vwap
            signal_df = calculate_vwap(signal_df)
            if not signal_df.empty:
                last_ts = signal_df.index[-1]
                try:
                    interval_min = int(interval_fyers)
                    # A candle is "Hybrid" (unclosed) if:
                    # 1. Its start time is not on an interval multiple OR
                    # 2. The current time is still within that candle's duration
                    current_time = pd.Timestamp.now(tz='Asia/Kolkata')
                    # Ensure last_ts is timezone-aware for comparison
                    if last_ts.tzinfo is None:
                        last_ts_aware = last_ts.tz_localize('Asia/Kolkata')
                    else:
                        last_ts_aware = last_ts.tz_convert('Asia/Kolkata')
                        
                    is_hybrid = False
                    if last_ts.minute % interval_min != 0:
                        is_hybrid = True
                    elif current_time < (last_ts_aware + pd.Timedelta(minutes=interval_min)):
                        # If we are before the next candle start, the current row is still live
                        is_hybrid = True
                        
                    if is_hybrid:
                        # logging.info(f"Hybrid Row Detected for {ticker} at {last_ts}. Excluding for Signal Trigger.")
                        signal_df = signal_df.iloc[:-1]
                except Exception as e:
                    logging.warning(f"Error in Hybrid Row Check: {e}")

            # --- STRATEGY & REGIME ---
            regime = "SNIPER" # Default
            signal_data = None

            
            if is_index:
                # === INDICES (5m, 9EMA) ===
                signal_df = calculate_ema(signal_df, period=9)
                signal_df = calculate_adx(signal_df)
                regime = detect_market_regime(signal_df)
                
                # HMM Regime Detection for Indices
                hmm_model = train_hmm_model(daily_df)
                hmm_regime = get_hmm_regime(hmm_model, daily_df)
                
                if len(signal_df) >= 5:
                    # Index: Cluster SL + 5 pts buffer
                    raw_signal = check_3_candle_setup(signal_df.iloc[-5:], sl_method='CLUSTER', limit_buffer=5.0, ema_col='EMA9')
                    strategy_name = f"3-Candle 9EMA ({regime}, HMM: {hmm_regime})"
                    
                    if raw_signal:
                        sig = raw_signal['Signal']
                        if hmm_regime == 'BULLISH' and sig == 'LONG':
                            signal_data = raw_signal
                        elif hmm_regime == 'BEARISH' and sig == 'SHORT':
                            signal_data = raw_signal
                        elif hmm_regime == 'CHOPPY' or hmm_regime == 'UNKNOWN':
                            # Strict blocking for indices: no trade in chop or uncertain regimes
                            logging.info(f"Skipping {sig} on {ticker} due to {hmm_regime} HMM Regime (Index Safety Filter)")
                        else:
                            logging.info(f"Skipping {sig} on {ticker} due to HMM Regime {hmm_regime}")
                    
                        # --- ADDITIONAL TREND FILTER: VWAP ---
                        if signal_data and 'VWAP' in signal_df.columns:
                            curr_c = signal_df.iloc[-1]
                            if sig == 'LONG' and curr_c['Close'] <= curr_c['VWAP']:
                                logging.info(f"Skipping LONG on {ticker}: Price {curr_c['Close']} is below VWAP {curr_c['VWAP']:.2f}")
                                signal_data = None
                            elif sig == 'SHORT' and curr_c['Close'] >= curr_c['VWAP']:
                                logging.info(f"Skipping SHORT on {ticker}: Price {curr_c['Close']} is above VWAP {curr_c['VWAP']:.2f}")
                                signal_data = None
                    
                current_price = intraday_df.iloc[-1]['Close']
                ema = signal_df.iloc[-1]['EMA9'] if 'EMA9' in signal_df.columns else current_price
            else:
                # === STOCKS (15m, 21EMA Strict) ===
                # Compute 21EMA for Stock
                signal_df = calculate_ema(signal_df, period=21)
                
                # HMM Regime Detection
                hmm_model = train_hmm_model(daily_df)
                hmm_regime = get_hmm_regime(hmm_model, daily_df)
                strategy_name = f"Stock 21EMA (HMM: {hmm_regime})"
                regime = "SNIPER" # Default to Sniper for stocks
                
                # Use HMM and ORB Breakdown
                if len(signal_df) >= 5:
                    
                    # CONFIG CHECK: ORB vs 9EMA
                    use_orb = config.get('ENABLE_ORB_STRATEGY', True) # Default to True for Stocks now
                    
                    if use_orb:
                        strategy_name = f"15m ORB (HMM: {hmm_regime})"
                        raw_signal = check_orb_breakout_setup(signal_df)
                        
                        if raw_signal:
                            if hmm_regime == 'BULLISH' and raw_signal['Signal'] == 'LONG':
                                signal_data = raw_signal
                            elif hmm_regime == 'BEARISH' and raw_signal['Signal'] == 'SHORT':
                                signal_data = raw_signal
                            else:
                                logging.info(f"Skipping {raw_signal['Signal']} on {ticker} due to HMM Regime {hmm_regime} (ORB requires strict alignment)")
                                
                    else:
                        strategy_name = f"Stock 21EMA (HMM: {hmm_regime})"
                        # Stock: Reference Candle High/Low SL + 0 buffer, 21EMA
                        raw_signal = check_3_candle_setup(signal_df.iloc[-5:], sl_method='REF', limit_buffer=0.0, ema_col='EMA21')
                        
                        if raw_signal:
                            # Filter signal based on HMM Regime for existing EMA strat
                            if hmm_regime == 'BULLISH' and raw_signal['Signal'] == 'LONG':
                                signal_data = raw_signal
                            elif hmm_regime == 'BEARISH' and raw_signal['Signal'] == 'SHORT':
                                signal_data = raw_signal
                            elif hmm_regime == 'CHOPPY' or hmm_regime == 'UNKNOWN':
                                # Allow both but SNIPER regime handles tight exits
                                signal_data = raw_signal
                            else:
                                logging.info(f"Skipping {raw_signal['Signal']} on {ticker} due to HMM Regime {hmm_regime}")
                                
                current_price = intraday_df.iloc[-1]['Close']
                ema = signal_df.iloc[-1]['EMA21'] if 'EMA21' in signal_df.columns else current_price

            if signal_data:
                # Deduplication Logic: Check if we already traded this Signal Time
                # We need to load full log now (df_log is available from earlier)
                signal_time = signal_data.get('Time')
                if signal_time:
                    # Convert to pd.Timestamp for comparison
                    try:
                        sig_ts = pd.to_datetime(signal_time)
                        if sig_ts.tzinfo is None:
                            sig_ts = sig_ts.tz_localize('Asia/Kolkata')
                        else:
                            sig_ts = sig_ts.tz_convert('Asia/Kolkata')

                        if not df_log.empty:
                            # Ensure df_log['Time'] is comparable
                            df_log['Time_Ts'] = pd.to_datetime(df_log['Time'])
                            # Ensure Time_Ts is also localized/converted
                            def safe_tz_convert(ts):
                                if ts.tzinfo is None:
                                    return ts.tz_localize('Asia/Kolkata')
                                return ts.tz_convert('Asia/Kolkata')
                            
                            df_log['Time_Ts'] = df_log['Time_Ts'].apply(safe_tz_convert)
                            
                            # Buffer: If we traded within 2 mins of this signal or after
                            existing = df_log[
                                (df_log['Ticker'] == ticker) & 
                                (df_log['Signal'] == signal_data['Signal']) &
                                (df_log['Time_Ts'] >= sig_ts) 
                            ]
                            if not existing.empty:
                                # logging.info(f"Skipping Duplicate: {ticker} Signal {sig_ts} already traded.")
                                continue
                    except Exception as e:
                        logging.warning(f"Dedupe Check Failed: {e}. Skipping trade for safety.")
                        continue
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
                # STRICT AI FOR INDICES: Never allow bypass for Nifty/BankNifty
                current_ai_bypass = False if is_index else ai_bypass
                
                ai_res = validator.validate_trade(ticker, signal_type, entry_p, technicals, allow_bypass=current_ai_bypass)
                ai_cooldowns[ticker] = ts
                
                if ai_res['valid']:
                    qty = 0
                    strike = 0
                    target_p = 0
                    order_sym = ""
                    
                    STOCK_OPTIONS_LOTS = {
                        "BHARTIARTL.NS": 475,
                        "DLF.NS": 825,
                        "TVSMOTOR.NS": 175,
                        "ASIANPAINT.NS": 250,
                        "HINDUNILVR.NS": 300,
                        "LT.NS": 175
                    }
                    is_stock_option = False
                    if not is_index and ticker in STOCK_OPTIONS_LOTS:
                        is_stock_option = True
                        
                    if is_index or is_stock_option:
                         # OPTION BUYING
                         otype = "CE" if signal_type == 'LONG' else "PE"
                         
                         if is_index:
                             # --- CONFIGURABLE LOT SIZE LOGIC FOR INDICES ---
                             TRADING_LOTS = config.get('TRADING_LOTS', 2) # Default 2 Lots
                             LOT_SIZE_NIFTY = config.get('LOT_SIZE_NIFTY', 65)
                             LOT_SIZE_BANKNIFTY = config.get('LOT_SIZE_BANKNIFTY', 30)

                             strike = int(get_strike_for_trade(entry_p, ticker, otype, depth=2)) 
                             lot_size = LOT_SIZE_BANKNIFTY if 'BANK' in ticker else LOT_SIZE_NIFTY
                             qty = int(TRADING_LOTS * lot_size)
                             logging.info(f"🔢 Qty Calculation for {ticker}: {TRADING_LOTS} Lots x {lot_size} = {qty}")
                         else:
                             # --- STOCK OPTIONS ---
                             # Default to trading 1 Lot for stocks due to high premium value
                             TRADING_LOTS_STOCKS = config.get('TRADING_LOTS_STOCKS', 1) 
                             lot_size = STOCK_OPTIONS_LOTS[ticker]
                             qty = int(TRADING_LOTS_STOCKS * lot_size)
                             strike = int(get_strike_for_trade(entry_p, ticker, otype, depth=0)) # ATM targeting for High Liquidity
                             logging.info(f"🔢 Stock Option Qty Calculation for {ticker}: {TRADING_LOTS_STOCKS} Lots x {lot_size} = {qty}")
                        
                         order_sym = get_fyers_symbol(ticker, strike, otype)
                         
                         # Dummy target so trailing logic runs purely on EMAs or HMM
                         risk_per_share = abs(entry_p - sl_p)
                         target_p = entry_p + (risk_per_share * 2 * (1 if signal_type == 'LONG' else -1)) if regime != "SURFER" else 0
                         
                    else:
                         # EQUITY
                         risk_per_share = abs(entry_p - sl_p)
                         qty = 1
                         if risk_per_share > 0:
                             risk_amt = ALLOCATION * RISK_PCT
                             qty = int(risk_amt / risk_per_share)
                         
                         # --- CRITICAL FIX: Cap Qty by Capital Allocation ---
                         # Ensure we don't exceed the max capital per trade
                         max_qty_cap = int(ALLOCATION / entry_p)
                         if qty > max_qty_cap:
                             qty = max_qty_cap
                             logging.info(f"Qty Capped by Capital: {qty} (Alloc: {ALLOCATION})")
                             
                         if qty < 1: qty = 1
                         
                         
                         order_sym = f"NSE:{ticker.replace('.NS', '')}-EQ"
                         target_p = entry_p + (risk_per_share * 2 * (1 if signal_type == 'LONG' else -1))

                    actual_entry = 0.0
                    if LIVE_TRADE and broker:
                        try:
                            # Fetch actual premium/price at entry for true PnL tracking
                            q = broker.get_quotes(order_sym)
                            if q and isinstance(q, list) and len(q) > 0 and 'v' in q[0]:
                                actual_entry = q[0]['v'].get('lp', 0.0)
                        except Exception as e:
                            logging.warning(f"Failed to fetch actual entry for {order_sym}: {e}")

                    full_signal = f"{signal_type} ({strategy_name})"
                    log_trade(ticker, order_sym, full_signal, entry_p, qty, sl_p, target_p, datetime.now(), ai_res['reason'], regime, actual_entry)
                    
                    if LIVE_TRADE and broker:
                        try:
                            side = "BUY"
                            if not (is_index or is_stock_option): side = "BUY" if signal_type == "LONG" else "SELL"
                            execute_order(order_sym, side, qty, 0)
                            
                            # --- CRITICAL FIX: Increment Counters Dynamically ---
                            if ticker == '^NSEI': nifty_trades += 1
                            elif ticker == '^NSEBANK': banknifty_trades += 1
                            else: 
                                stock_trades += 1 # Global stock trade count
                                # Note: Per-stock limit (MAX_TRADES_PER_STOCK) is checked at start of loop
                                # but we should also update 'tod' or track it? 
                                # Since we re-read 'tod' at start of check_for_signals, we just need to ensure
                                # we don't take ANOTHER trade for THIS stock in THIS loop pass (which we won't as loop moves to next).
                                # But 'stock_trades' is global limit, so we must increment it to stop subsequent stocks if limit reached.
                            
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
                intraday = calculate_ema(intraday, period=9)
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
                    
                # 3. Check Profit Booking / Strategy Exit
                # Depends on REGIME
                
                target_hit = False
                strategy_exit = False
                
                if regime == 'SNIPER':
                    # SNIPER MODE: Fixed Target Exit
                    # If Long: Price >= Target
                    # If Short: Price <= Target
                    target_p = float(trade.get('Target', 0))
                    
                    if target_p > 0:
                        if 'LONG' in signal and current_price >= target_p:
                            target_hit = True
                            exit_msg = "Target Hit (Sniper)"
                            target_hit = True
                            exit_msg = "Target Hit (Sniper)"
                        elif 'SHORT' in signal and current_price <= target_p:
                            target_hit = True
                            exit_msg = "Target Hit (Sniper)"
                            
                    # HYBRID EXIT: Check for 9EMA Reversal (Safety)
                    # Exit if Previous Candle CLOSED on opposite side of 9EMA
                    if not target_hit:
                        ema_cross = False
                        if 'LONG' in signal and prev['Close'] < prev['EMA9']:
                            ema_cross = True
                            exit_msg = "Trailing Exit (Close < 9EMA)"
                        elif 'SHORT' in signal and prev['Close'] > prev['EMA9']:
                            ema_cross = True
                            exit_msg = "Trailing Exit (Close > 9EMA)"
                            
                        if ema_cross:
                            logging.info(f"🌊 9EMA TRAILING EXIT for {ticker}: {current_price} vs 9EMA {prev['EMA9']:.2f}")
                            close_trade_execution(idx, ticker, instrument, qty, signal, current_price, exit_msg, pnl)
                            continue
                            
                    if target_hit:
                        logging.info(f"🎯 TARGET HIT for {ticker}: {current_price} vs {target_p}")
                        close_trade_execution(idx, ticker, instrument, qty, signal, current_price, exit_msg, pnl)
                        continue

                elif regime == 'SURFER':
                    # SURFER MODE: 9EMA Trailing Exit on CLOSED Candle
                    # Long Exit: Close < 9EMA
                    # Short Exit: Close > 9EMA
                    
                    if 'LONG' in signal:
                        if prev['Close'] < prev['EMA9']:
                            strategy_exit = True
                            exit_msg = "9EMA Reversal (Close < 9EMA)"
                    else:
                        if prev['Close'] > prev['EMA9']:
                            strategy_exit = True
                            exit_msg = "9EMA Reversal (Close > 9EMA)"
                            
                    if strategy_exit:
                        logging.info(f"🌊 SURFER EXIT for {ticker}: {prev['Close']} vs EMA {prev['EMA9']:.2f}")
                        close_trade_execution(idx, ticker, instrument, qty, signal, current_price, exit_msg, pnl)
                        continue
                    
            except Exception as e:
                logging.error(f"Monitor Error {ticker}: {e}")

    except Exception as e:
        logging.error(f"Monitor Loop Error: {e}")

def close_trade_execution(idx, ticker, instrument, qty, signal, price, reason, pnl):
    # 2. Execute Market Order FIRST
    order_success = False
    if LIVE_TRADE and broker:
        try:
            side = "SELL" # Default for Closing Long
            if 'SHORT' in signal: side = "BUY" # Closing Short
            
            # CRITICAL FIX: If Option (CE/PE), we always BUY to Open, so always SELL to Close
            if "CE" in instrument or "PE" in instrument:
                side = "SELL"
            
            # For Indices/Fyers Options, we Buy/Sell the Instrument
            order_success = execute_order(instrument, side, qty, 0)
            
            if not order_success:
                 logging.error(f"❌ EXIT FAILED for {ticker}. Check Broker! Trade NOT Closed in DB.")
                 return # DO NOT CLOSE IN DB
                 
        except Exception as e:
            logging.error(f"Close Execution Failed: {e}")
            return

    # 1. Log Exit ONLY if Order Success (or Paper Mode)
    close_trade(idx, price, reason, pnl)
    logging.info(f"✅ CLOSED {ticker}: {reason} | PnL: {pnl:.2f}")

def close_trade(idx, price, reason, pnl):
    try:
        df = pd.read_csv(TRADE_LOG)
        
        # Calculate Charges
        entry_p = float(df.at[idx, 'Entry_Price'])
        qty = int(df.at[idx, 'Qty'])
        ticker = str(df.at[idx, 'Ticker'])
        instrument = str(df.at[idx, 'Instrument'])
        is_option = 'CE' in instrument or 'PE' in instrument
        
        actual_entry = df.at[idx, 'Actual_Entry'] if 'Actual_Entry' in df.columns else 0.0
        actual_exit = 0.0
        
        if LIVE_TRADE and broker:
             try:
                 q = broker.get_quotes(instrument)
                 if q and isinstance(q, list) and len(q) > 0 and 'v' in q[0]:
                     actual_exit = q[0]['v'].get('lp', 0.0)
             except Exception as e:
                 logging.warning(f"Failed to fetch actual exit for {instrument}: {e}")
                 
        # Override PNL if we have Actuals
        if actual_entry > 0 and actual_exit > 0:
             if 'SHORT' in str(df.at[idx, 'Signal']) and not is_option:
                  pnl = (actual_entry - actual_exit) * qty
             else:
                  pnl = (actual_exit - actual_entry) * qty

        charges = calculate_charges(entry_p, price, qty, ticker, is_option)
        net_pnl = pnl - charges
        
        df.at[idx, 'Status'] = 'CLOSED'
        df.at[idx, 'Exit_Time'] = datetime.now()
        df.at[idx, 'Exit_Price'] = price
        df.at[idx, 'Exit_Reason'] = reason
        df.at[idx, 'PnL'] = pnl
        df.at[idx, 'Charges'] = charges
        df.at[idx, 'Net_PnL'] = net_pnl
        df.at[idx, 'Actual_Exit'] = actual_exit
        
        df.to_csv(TRADE_LOG, index=False)
    except: pass

def log_trade(ticker, instrument, signal, price, qty, sl, target, time, notes="", regime="SNIPER", actual_entry=0.0):
    logging.info(f"SIGNAL: {signal} [{regime}] {qty} x {ticker} ({instrument}) @ {price} | SL {sl:.2f}")
    if not os.path.exists(TRADE_LOG):
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(TRADE_LOG, index=False)
    
    new_data = {
        'Time': time, 'Ticker': ticker, 'Instrument': instrument, 'Signal': signal, 'Entry_Price': price,
        'Qty': qty, 'SL': sl, 'Target': target, 'Notes': notes, 'Status': 'OPEN', 'PnL': 0.0, 'Regime': regime,
        'Charges': 0.0, 'Net_PnL': 0.0, 'Actual_Entry': actual_entry, 'Actual_Exit': 0.0
    }
    pd.DataFrame([new_data], columns=TRADE_COLUMNS).to_csv(TRADE_LOG, mode='a', header=False, index=False)

def execute_order(symbol, side, qty, stop_loss):
    if not broker: return False
    try:
        logging.info(f"🚀 LIVE ORDER: {side} {qty} {symbol}")
        res = broker.place_order(symbol, int(qty), side, order_type="MARKET", product_type="INTRADAY", limit_price=0, stop_price=0)
        
        # Fyers API v3 returns 'id' on success
        if 'id' in res: 
            logging.info(f"Order Placed: {res['id']}")
            return True
        else:
            logging.error(f"Order Rejected/Failed: {res}")
            return False
            
    except Exception as e: 
        logging.error(f"Execution Error: {e}")
        return False

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
    global FYERS_TOKEN, LIVE_TRADE, broker
    logging.info("--- Live Bot (Indices + Stocks + ADX Dynamic) Started ---")
    watchlist = get_watchlist()
    # Initialize Futures Bot
    f_bot = PositionalFuturesBot(broker) if LIVE_TRADE and broker else PositionalFuturesBot(None)
    
    while True:
        try:
            now = datetime.now()
            
            # Pre-Market
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                 logging.info(f"Pre-Market Wait... {now.strftime('%H:%M:%S')}")
                 time.sleep(60)
                 continue
                 
            # Post-Market / Square Off (3:15 PM) -> ONLY for Intraday!
            # Futures are Positional, so we DO NOT square them off here.
            
            if now.hour >= 15 and now.minute >= 15:
                 logging.info("Market Closing (3:15 PM). Squaring Off Intraday Positions...")
                 square_off_all_positions() 
                 # We continue the loop? Or sleep until next day?
                 # Actually `square_off_all_positions` handles only Intraday via Trade Log.
                 # Futures state is separate.
                 # Strategy says: Check at Daily Close. 3:15-3:29 is good time to check.
                 
            # Reload config
            config = load_config() 
            
            # Dynamic Token Reload for seamless Re-Auth
            load_dotenv(override=True)
            new_token = os.environ.get("FYERS_TOKEN")
            
            # Check if Token Changed or Broker is Disconnected
            if new_token and (new_token != FYERS_TOKEN or broker is None):
                logging.info("🔄 Token Changed/Found! Re-Initializing Broker...")
                FYERS_TOKEN = new_token
                try:
                    broker = FyersApp(FYERS_CLIENT_ID, "secret_placeholder", access_token=FYERS_TOKEN)
                    profile = broker.get_profile()
                    if 'error' in profile or profile.get('s') == 'error':
                         logging.error(f"❌ Re-Auth Failed: {profile}")
                         LIVE_TRADE = False
                    else:
                         logging.info(f"✅ Re-Auth SUCCEEDED: {profile.get('data', {}).get('name', 'User')}")
                         LIVE_TRADE = True
                         # Update Futures Bot too
                         f_bot = PositionalFuturesBot(broker)
                except Exception as e:
                    logging.error(f"❌ Re-Auth Exception: {e}")
                    
            monitor_positions() # Check existing trades FIRST
            check_for_signals(watchlist, config)
            
            # --- POSITIONAL FUTURES CHECK (Daily Candle) ---
            # Check every loop? Daily candles update real-time.
            # Strategy says "When Daily candle closes...". 
            # Ideally check near close (e.g. > 3:15 PM) OR continuously if we want to enter mid-day?
            # User said "Daily time frame... When Daily candle closes".
            # Let's check generally, `pos_futures.py` uses `iloc[-1]` (current) or `iloc[-2]` (confirmed).
            # If we use `iloc[-2]`, we are acting on YESTERDAY'S close.
            # If we want to act TODAY, we need to check `iloc[-1]` near market close (3:20 PM).
            # For now, let's run it every loop but `pos_futures` logic is crucial.
            # Updated `pos_futures` to look at `prev` (confirmed).
            # So this enters NEXT MORNING.
            # IF user wants SAME DAY ENTRY, we need to look at `curr` near 3:25 PM.
            # Let's run it.
            
            if config.get("ENABLE_FUTURES_STRATEGY", False):
                futures_tickers = ["^NSEI"] # Only Indices? "only for indices futures"
                # Add ^NSEBANK if needed, user said "Indices Futures".
                futures_tickers.append("^NSEBANK")
                
                for ft in futures_tickers:
                    try:
                        # Fetch Daily Data
                        daily, _ = fetch_data(ft, broker=broker) # uses caching
                        if daily.empty: continue
                        
                        sym_future = get_futures_symbol(ft)
                        
                        # Calculate Lot Size
                        # Use same config as Options?
                        # Nifty Future Lot = 75 (or 65? It changes). 
                        # Fyers API might give lot size? 
                        # For now use config integers.
                        lot_size = config.get('LOT_SIZE_BANKNIFTY', 30) if 'BANK' in ft else config.get('LOT_SIZE_NIFTY', 65)
                        trading_lots = config.get('TRADING_LOTS', 2)
                        
                        action = f_bot.execute_logic(ft, daily, lot_size, trading_lots, sym_future)
                        
                        if action:
                             # EXECUTE REAL ORDER
                             if LIVE_TRADE and broker:
                                 # action = {signal, symbol, qty, reason, sl}
                                 qty = action['qty']
                                 # Buy or Sell?
                                 # Signal: LONG, SHORT, REVERSE_LONG, REVERSE_SHORT
                                 
                                 if "REVERSE" in action['signal']:
                                     # Close opposite first
                                     # Need to know opposite qty?
                                     # Simplified: Just place the NEW order for now, 
                                     # assuming manual intervention for reversal or sophisticated logic later.
                                     # Wait, reversal means Exit Old + Enter New.
                                     # Currently Execute Logic just returns "What to do NOW".
                                     # If it returns LONG, we Buy.
                                     pass
                                     
                                 side = "BUY" if "LONG" in action['signal'] else "SELL"
                                 logging.info(f"🚀 EXECUTING PUTURES: {side} {action['symbol']} x {qty}")
                                 execute_order(action['symbol'], side, qty, 0) # Market Order
                                 
                    except Exception as e:
                        logging.error(f"Futures Loop Error {ft}: {e}")

            time.sleep(60)
        except KeyboardInterrupt: break
        except Exception as e:
            logging.error(f"Loop error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
