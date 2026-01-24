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

if __name__ == "__main__":
    pass
