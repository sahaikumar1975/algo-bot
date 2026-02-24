from live_bot import get_strike_for_trade
from option_util import get_fyers_symbol
import datetime

def test_options_setup():
    test_cases = [
        {"ticker": "LT.NS", "price": 3120, "otype": "CE"},
        {"ticker": "LT.NS", "price": 3120, "otype": "PE"},
        {"ticker": "BHARTIARTL.NS", "price": 1982, "otype": "CE"},
        {"ticker": "BHARTIARTL.NS", "price": 1982, "otype": "PE"},
    ]
    
    print("--- Options Setup Verification ---")
    for tc in test_cases:
        strike = get_strike_for_trade(tc['price'], tc['ticker'], tc['otype'], depth=0)
        # We need to simulate get_fyers_symbol which expects ticker without .NS or handled in util
        symbol = get_fyers_symbol(tc['ticker'], strike, tc['otype'])
        print(f"Ticker: {tc['ticker']} | Spot: {tc['price']} | Signal: {tc['otype']}")
        print(f"Calculated Strike: {strike} | Generated Symbol: {symbol}")
        print("-" * 30)

if __name__ == "__main__":
    test_options_setup()
