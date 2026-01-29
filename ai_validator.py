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

    def validate_trade(self, ticker: str, signal: str, price: float, technicals: Dict[str, Any]) -> Dict[str, Any]:
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
            # We use gemini-2.0-flash if available, or 1.5-flash
            response = self.client.models.generate_content(
                model='gemini-2.0-flash', 
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
            logging.error(f"AI Validation Error: {e}")
            # Fail safe: Allow trade but log warning
            return {'valid': True, 'reason': f'AI Error ({str(e)}) - Defaulting to Valid', 'mode': 'SNIPER'}

if __name__ == "__main__":
    # Test
    # export GEMINI_API_KEY='your_key'
    pass
