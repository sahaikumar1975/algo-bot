"""
AI Validator Module
-------------------
Uses Google's Gemini Flash model to validate trade setups based on technical data.
"""

import os
import logging
import json
try:
    from google import genai
except ImportError:
    genai = None

from typing import Dict, Any
from datetime import datetime
import numpy as np

class AIValidator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                if genai:
                    self.client = genai.Client(api_key=self.api_key)
                else:
                    logging.error("google-genai package not found. Please pip install google-genai")
            except Exception as e:
                logging.error(f"Failed to initialize Gemini AI: {e}")

    def log_usage(self):
        """Log the AI call timestamp to a CSV file."""
        try:
            log_file = "ai_usage_log.csv"
            # Use absolute path relative to this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            log_path = os.path.join(base_dir, log_file)
            
            with open(log_path, "a") as f:
                # Format: Timestamp
                f.write(f"{datetime.now().isoformat()}\n")
        except Exception as e:
            logging.error(f"Failed to log AI usage: {e}")

    def validate_trade(self, ticker: str, signal: str, price: float, technicals: Dict[str, Any], allow_bypass: bool = False) -> Dict[str, Any]:
        """
        Asks Gemini AI to validate the trade.
        Returns: {'valid': bool, 'reason': str}
        """
        if not self.client:
            return {'valid': True, 'reason': 'AI Validation Disabled (No API Key or Package Missing)'}

        def default(o):
            if isinstance(o, (np.integer, np.int64)): return int(o)
            if isinstance(o, (np.floating, np.float64)): return float(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        prompt = f"""
        You are a professional day trader. Validate this trade setup and recommend a Strategy Mode.


        Ticker: {ticker}
        Signal: {signal}
        Entry Price: {price}
        
        
        Technical Context:
        {json.dumps(technicals, indent=2, default=default)}
        
        Task:
        1. Validate the trade (High Probability?).
        2. Choose Strategy Mode:
           - "SNIPER": For Choppy/Range-Bound/Weak Trend markets. (Uses Fixed 1:2 Target).
           - "SURFER": For Strong Trend/Breakout markets. (Uses EMA 20 Trailing Stop).
        
        Output format (JSON only):
        {{
            "valid": true/false,
            "reason": "Short explanation...",
            "mode": "SNIPER" or "SURFER"
        }}
        """
        
        try:
            # Using the new SDK syntax
            # We use gemini-flash-latest based on available models
            
            # Log usage before call
            self.log_usage()

            response = self.client.models.generate_content(
                model='gemini-2.0-flash-lite-001', 
                contents=prompt
            )
            
            # Clean response
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:-3]
            
            result = json.loads(text)
            return {
                'valid': result.get('valid', False),
                'reason': result.get('reason', 'AI Error'),
                'mode': result.get('mode', 'SNIPER') # Default to Sniper
            }
        except Exception as e:
            error_msg = str(e)
            logging.error(f"AI Validation Error: {error_msg}")
            
            if "429" in error_msg or "Resource has been exhausted" in error_msg:
                 if allow_bypass:
                     return {
                         'valid': True, # SOFT FAIL: Allow trade despite AI Limit
                         'reason': 'AI Rate Limit Exceeded (429) - Bypassed',
                         'mode': 'SNIPER'
                     }
                 else:
                     return {
                         'valid': False, # STRICT FAIL: Block trade
                         'reason': 'AI Rate Limit Exceeded (429) - Blocked',
                         'mode': 'SNIPER'
                     }
            
            # CRITICAL AUTH ERRORS (400, 401, 403)
            # 400: Bad Request (Expired Key)
            # 401: Unauthorized
            # 403: Forbidden
            if "400" in error_msg or "401" in error_msg or "403" in error_msg or "API key expired" in error_msg:
                if allow_bypass:
                     return {
                         'valid': True, # SOFT FAIL: Allow trade despite Auth Error
                         'reason': f'AI Auth Error ({error_msg}) - Bypassed',
                         'mode': 'SNIPER'
                     }
                return {
                    'valid': False, # STRICT FAIL SAFE
                    'reason': f'AI CRITICAL ERROR: {error_msg} - CHECK API KEY',
                    'mode': 'SNIPER'
                }
            
            # Fail safe: Allow trade but log warning (for other errors like timeouts or random 500s)
            return {'valid': True, 'reason': f'AI Error ({error_msg}) - Defaulting to Valid', 'mode': 'SNIPER'}

if __name__ == "__main__":
    # Test
    # export GEMINI_API_KEY='your_key'
    pass
