import logging
import time
import sys
from fyers_integration import FyersApp

# Setup Logging to Console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Mock Broker Class
class MockBroker:
    def __init__(self):
        self.orders = {}
        self.order_counter = 1000

    def place_order(self, symbol, qty, side, order_type='MARKET', product_type='INTRADAY', limit_price=0, stop_price=0):
        logging.info(f"[MOCK BROKER] Placing Order: {side} {qty} {symbol} ({order_type})")
        
        # Simulate API Response
        order_id = f"TEST_ORD_{self.order_counter}"
        self.order_counter += 1
        
        # Store order details for verification
        self.orders[order_id] = {
            "status": 1, # Pending
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "price": 0.0 # Not filled yet
        }
        
        # Simulate Fill Delay (Async)
        import threading
        def fill_order():
            time.sleep(2) # 2 second delay
            self.orders[order_id]["status"] = 2 # Filled
            self.orders[order_id]["price"] = 245.50 if "CE" in symbol else 100.0
            logging.info(f"[MOCK BROKER] Order {order_id} FILLED internally.")

        threading.Thread(target=fill_order).start()
        
        return {"s": "ok", "code": 1101, "message": "Order Submitted", "id": order_id}

    def verify_order_status(self, order_id, max_retries=10):
        logging.info(f"[MOCK BROKER] Verifying Order {order_id}...")
        for i in range(max_retries):
            if order_id in self.orders:
                status = self.orders[order_id]["status"]
                if status == 2:
                    price = self.orders[order_id]["price"]
                    return "FILLED", price
            time.sleep(1)
        return "TIMEOUT", 0.0

# Import live_bot
import live_bot

# Inject Mock Broker
logging.info("--- STARTING TEST: Injecting Mock Broker ---")
live_bot.broker = MockBroker()

# Define Test Case
test_ticker = "^NSEI" # Nifty
test_signal = "LONG [24500 CE]" # Mock Signal
test_qty = 50 # Will be adjusted to 65 (Lot Size)
test_sl = 0 # Not used for logic, calculated dynamically

logging.info(f"--- TRIGGERING TEST SIGNAL: {test_signal} ---")

# Call Execution
live_bot.execute_order(test_ticker, test_signal, test_qty, test_sl)

logging.info("--- TEST COMPLETE ---")
