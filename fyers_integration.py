"""Fyers API V3 Wrapper.

Handles Authentication (Auth Code -> Token) and Order Execution.
"""
import os
import base64
import requests
from fyers_apiv3 import fyersModel
import pandas as pd
from datetime import datetime

# Default Redirect URI (Commonly used)
DEFAULT_REDIRECT_URI = "https://trade.fyers.in/api-login/redirect-uri/index.html"

class FyersApp:
    def __init__(self, client_id, secret_key, redirect_uri=DEFAULT_REDIRECT_URI, access_token=None, log_path="."):
        self.client_id = client_id
        self.secret_key = secret_key
        self.redirect_uri = redirect_uri
        self.access_token = access_token
        self.fyers = None
        self.log_path = log_path

        if access_token:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the main FyersModel with the token."""
        try:
            self.fyers = fyersModel.FyersModel(
                client_id=self.client_id,
                is_async=False,
                token=self.access_token,
                log_path=self.log_path
            )
        except Exception as e:
            print(f"Error initializing FyersModel: {e}")

    def get_login_url(self):
        """Generate the Login URL for the user."""
        try:
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type='code',
                grant_type='authorization_code'
            )
            return session.generate_authcode()
        except Exception as e:
            return f"Error generating URL: {e}"

    def generate_access_token(self, auth_code):
        """Exchange Auth Code for Access Token."""
        try:
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_uri,
                response_type='code',
                grant_type='authorization_code'
            )
            session.set_token(auth_code)
            response = session.generate_token()
            
            if 'access_token' in response:
                self.access_token = response['access_token']
                self._initialize_model()
                return self.access_token
            else:
                raise Exception(f"Token generation failed: {response}")
        except Exception as e:
            raise Exception(f"Error in generate_access_token: {e}")

    def get_profile(self):
        """Fetch User Profile to validate token."""
        if not self.fyers: return None
        return self.fyers.get_profile()

    def place_order(self, symbol, qty, side, order_type='MARKET', product_type='INTRADAY', limit_price=0, stop_price=0):
        """
        Place Order via V3.
        side: 1 (Buy) or -1 (Sell)
        type: 1 (Limit), 2 (Market), 3 (SL-M), 4 (SL-L)
        """
        if not self.fyers: return {"s": "error", "message": "Fyers not initialized"}

        # Map string inputs to Fyers Constants
        f_side = 1 if str(side).upper() == 'BUY' else -1
        
        f_type = 2 # Market default
        if order_type == 'LIMIT': f_type = 1
        elif order_type == 'SL-M': f_type = 3
        elif order_type == 'SL-L': f_type = 4
        
        f_product = product_type # 'INTRADAY', 'CNC', 'MARGIN'

        data = {
            "symbol": symbol,
            "qty": int(qty),
            "type": f_type,
            "side": f_side,
            "productType": f_product,
            "limitPrice": float(limit_price),
            "stopPrice": float(stop_price),
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
        }

        try:
            response = self.fyers.place_order(data=data)
            return response
        except Exception as e:
            return {"s": "error", "message": str(e)}

    def verify_order_status(self, order_id, max_retries=10):
        """
        Polls order status until it is FILLED or Rejected/Cancelled.
        Returns: (status, filled_price)
        """
        if not self.fyers: return "ERROR", 0.0
        
        import time
        for _ in range(max_retries):
            try:
                data = {"id": order_id}
                res = self.fyers.orderbook(data=data)
                
                if 'orderBook' in res and len(res['orderBook']) > 0:
                    order = res['orderBook'][0]
                    status = order.get('status') # 2=Filled, 6=Cancelled, 5=Rejected
                    
                    if status == 2: # FILLED
                        # Try to get average traded price
                        price = float(order.get('tradedPrice', 0))
                        return "FILLED", price
                        
                    elif status in [5, 6]:
                        return "REJECTED", 0.0
                
                time.sleep(1) # Wait 1s before next check
            except Exception as e:
                print(f"Order Check Error: {e}")
                
        return "TIMEOUT", 0.0

    def get_positions(self):
        if not self.fyers: return pd.DataFrame()
        try:
            res = self.fyers.positions()
            if 'netPositions' in res:
                return pd.DataFrame(res['netPositions'])
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return pd.DataFrame()

    def get_quotes(self, symbols):
        """
        Fetch Real-Time Quotes for a list of symbols.
        symbols: Comma separated string or list of symbols (e.g. "NSE:NTPC-EQ,NSE:SBIN-EQ")
        """
        if not self.fyers: return None
        try:
            # Format input
            if isinstance(symbols, list):
                symbols = ",".join(symbols)
            
            data = {"symbols": symbols}
            response = self.fyers.quotes(data=data)
            
            if 'd' in response:
                return response['d'] # List of dicts
            return None
        except Exception as e:
            print(f"Error fetching quotes: {e}")
            return None

    def get_history(self, symbol, resolution, range_from, range_to, date_format='0'):
        """
        Fetch Historical Data.
        resolution: 'D', '1D', '5'
        range_from/to: 'yyyy-mm-dd'
        """
        if not self.fyers: return None
        try:
            # Universal Symbol Converter (yfinance -> Fyers)
            fy_symbol = symbol
            if not "NSE:" in symbol and not "BSE:" in symbol:
                clean_t = symbol.replace('.NS', '')
                if clean_t.startswith('^') or clean_t == 'NSEI' or clean_t == 'NSEBANK': 
                     if 'NSEI' in clean_t: fy_symbol = "NSE:NIFTY50-INDEX"
                     elif 'BANK' in clean_t: fy_symbol = "NSE:NIFTYBANK-INDEX"
                else:
                    fy_symbol = f"NSE:{clean_t}-EQ"
            
            data = {
                "symbol": fy_symbol,
                "resolution": str(resolution),
                "date_format": str(date_format),
                "range_from": range_from,
                "range_to": range_to,
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data=data)
            
            if 'candles' in response:
                df = pd.DataFrame(response['candles'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                
                # Convert Date (Epoch or String) to Datetime
                if date_format == '1': # Epoch
                     df['Date'] = pd.to_datetime(df['Date'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
                else:
                     # If string, might need parsing depending on Fyers return format. 
                     # Usually Fyers returns Epoch even if requested otherwise in some versions, 
                     # but let's assume epoch response for safety if valid. 
                     # Actually standard v3 returns list of lists.
                     pass 
                
                # Ensure Timestamp Index for yfinance compatibility
                df['Date'] = pd.to_datetime(df['Date'], unit='s', utc=True).dt.tz_convert('Asia/Kolkata')
                df.set_index('Date', inplace=True)
                return df
                
            return pd.DataFrame() # Empty if no candles
            
        except Exception as e:
            print(f"Error fetching history for {symbol}: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    pass
