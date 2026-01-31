import os
import logging
from fyers_integration import FyersApp
from dotenv import load_dotenv

# Load Env
load_dotenv("/Users/sahaikumar/Projects/SMA2150/.env")

# Config
FYERS_TOKEN = os.environ.get("FYERS_TOKEN")
FYERS_CLIENT_ID = os.environ.get("FYERS_CLIENT_ID")
TICKER_SYMBOL = "NSE:KOTAKBANK-EQ"
QTY = 1

logging.basicConfig(level=logging.INFO)

def test_trade():
    print("--- LIVE TRADE TEST (KOTAKBANK) ---")
    
    if not FYERS_TOKEN or not FYERS_CLIENT_ID:
        print("❌ ERROR: FYERS_TOKEN or CLIENT_ID missing in .env")
        return

    try:
        print("Connecting to Fyers...")
        broker = FyersApp(FYERS_CLIENT_ID, "secret_placeholder", access_token=FYERS_TOKEN)
        profile = broker.get_profile()
        print(f"✅ Connected: {profile}")
        
        # Confirm
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == 'yes':
             print(f"⚠️  WARNING: Auto-confirming LIVE TRADE for {QTY} x {TICKER_SYMBOL}...")
        else:
            confirm = input(f"⚠️  WARNING: Using LIVE MONEY to buy {QTY} x {TICKER_SYMBOL}. Proceed? (y/n): ")
            if confirm.lower() != 'y':
                print("Aborted.")
                return

        print(f"🚀 Placing Order: BUY {QTY} {TICKER_SYMBOL} Market...")
        res = broker.place_order(TICKER_SYMBOL, QTY, "BUY", order_type="MARKET", product_type="INTRADAY")
        
        print(f"📝 Response: {res}")
        
        if 'id' in res:
            print("✅ SUCCESS! Order Placed.")
        else:
            print("❌ FAILED. Response indicates error.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_trade()
