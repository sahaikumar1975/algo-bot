"""
AI Validator Module
-------------------
Uses Google's Gemini Flash model to validate trade setups based on technical data.
"""

import os
import logging
import json
import google.generativeai as genai
from typing import Dict, Any

class AIValidator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception as e:
                logging.error(f"Failed to initialize Gemini AI: {e}")

    def validate_trade(self, ticker: str, signal: str, price: float, technicals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asks Gemini AI to validate the trade.
        Returns: {'valid': bool, 'reason': str}
        """
        if not self.model:
            return {'valid': True, 'reason': 'AI Validation Disabled (No API Key)'}

        prompt = f"""
        You are a professional day trader. Validate this trade setup and recommend a Strategy Mode.

        Ticker: {ticker}
        Signal: {signal}
        Entry Price: {price}
        
        Technical Context:
        {json.dumps(technicals, indent=2)}
        
        Task:
        1. Validate the trade (High Probability?).
        2. Choose Strategy Mode:
           - "SNIPER": For Choppy/Range-Bound/Weak Trend markets. (Uses Fixed 1:2 Target).
           - "SURFER": For Strong Trend/Breakout markets (Narrow CPR, High Vol). (Uses EMA 20 Trailing Stop).
        
        Output format (JSON only):
        {{
            "valid": true/false,
            "reason": "Short explanation...",
            "mode": "SNIPER" or "SURFER"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Clean response to ensure it's valid JSON
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
