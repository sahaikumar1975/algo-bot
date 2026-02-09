
import os
import logging
import datetime
from dotenv import load_dotenv
from fyers_integration import FyersApp
from option_util import get_next_expiry, get_fyers_symbol

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Env
load_dotenv()
client_id = os.getenv("FYERS_CLIENT_ID")
secret_key = os.getenv("FYERS_SECRET")
access_token = os.getenv("FYERS_TOKEN")

if not access_token:
    logging.error("No Access Token found in .env")
    exit()

# Initialize Fyers
broker = FyersApp(client_id, secret_key, access_token=access_token)
profile = broker.get_profile()
if 's' in profile and profile['s'] == 'error':
    logging.error(f"Fyers Auth Failed: {profile}")
    exit()

logging.info(f"✅ Fyers Connected: {profile.get('data', {}).get('name', 'Unknown')}")

# Get Current Nifty Price
quotes = broker.get_quotes("NSE:NIFTY50-INDEX")
if not quotes:
    logging.error("Failed to fetch Nifty Price")
    exit()

nifty_ltp = quotes[0].get('v', {}).get('lp', 0)
logging.info(f"Current Nifty Price: {nifty_ltp}")

# 1. Calculate Strike (ATM or Slightly ITM)
# Round to nearest 50
strike = int(round(nifty_ltp / 50) * 50)
logging.info(f"Selected Strike: {strike}")

# 2. Brute Force Symbol Discovery (Round 4 - Verbose)
logging.info("🔎 Probing with Connectivity Check & Alternative Bases...")

# Step A: Verify Basic Connectivity via INDEX (Known working)
q_index = broker.get_quotes("NSE:NIFTY50-INDEX")

# broker.get_quotes returns the LIST of quotes (data['d'])
if q_index and isinstance(q_index, list) and len(q_index) > 0:
    logging.info(f"✅ Connectivity OK: NIFTY50 Quote: {q_index[0].get('v', {}).get('lp', 'N/A')}")
else:
    logging.error(f"❌ Connectivity Check Failed (NIFTY50-INDEX). Response: {q_index}")
    exit()

# Step B: Probe Nifty Options
# Feb 2026: 12 (Thu)
d = 12
m = 2
year = 26
strike = 25600 

bases = ["NIFTY", "NIFTY50", "NIFTY 50", "FINNIFTY", "BANKNIFTY"] # Added others just in case
date_formats = [
    (f"{year}FEB{d}", "YYMMMdd"), # 26FEB12 (Original)
    (f"{year}2{d:02d}", "YYMdd"), # 26212 (Fyers Standard?)
    (f"{year}02{d:02d}", "YYMMdd"), # 260212 (Alt)
    (f"{year}FEB", "YYMMM"), # 26FEB (Monthly)
    (f"{year}MAR{d}", "YYMMMdd (March?)"), # Wrong month check?
]

valid_symbol = None

for base in bases:
    for dfmt, label in date_formats:
        # Try ATM (25600) and Next (25650)
        for s in [25600, 25650]:
            cand = f"NSE:{base}{dfmt}{s}CE"
            logging.info(f"Checking: {cand}")
            try:
                q = broker.get_quotes(cand)
                
                # Check for list response
                if q and isinstance(q, list) and len(q) > 0:
                    first = q[0]
                    if 's' in first and first['s'] == 'error':
                        logging.warning(f"  -> Error Response for {cand}: {first}")
                        continue
                        
                    # Valid quote structure, but check if price exists
                    lp = first.get('v', {}).get('lp', 0)
                    if lp > 0:
                        logging.info(f"✅ FOUND TRADABLE SYMBOL: {cand}")
                        logging.info(f"Quote: {lp}")
                        valid_symbol = cand
                        break
                    else:
                        logging.warning(f"⚠️  Symbol {cand} found but no Liquid Price (lp={lp})")
            except Exception as e:
                logging.error(f"Quote Error {cand}: {e}")
                
        if valid_symbol: break
    if valid_symbol: break

if not valid_symbol:
    logging.error("Could not find any valid symbol format after extended probe.")
    exit()

logging.info(f"🚀 Triggering LIVE BUY Order for: {valid_symbol}")
logging.info("Quantity: 65 (1 Lot)") 
qty = 65 

# confirmation
confirm = input(f"Confirm BUY {qty} x {valid_symbol} at MARKET? (y/n): ")
if confirm.lower() != 'y':
    print("Aborted.")
    exit()

response = broker.place_order(valid_symbol, qty, "BUY", order_type="MARKET", product_type="INTRADAY")
logging.info(f"Order Response: {response}")
