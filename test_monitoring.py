
import pandas as pd
import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Mock Broker
sys.modules['fyers_integration'] = MagicMock()

# Import bot functions (will need to mock logging and constants)
with patch('live_bot.load_config', return_value={}):
    import live_bot
    from live_bot import monitor_positions

class TestMonitoring(unittest.TestCase):
    
    def setUp(self):
        # Setup dummy trade log
        self.trade_log = "test_trade_log.csv"
        live_bot.TRADE_LOG = self.trade_log
        
        # Create a dummy open trade
        df = pd.DataFrame([{
            'Time': '2025-01-01 10:00:00',
            'Ticker': 'MOCK.NS',
            'Instrument': 'MOCK-EQ',
            'Signal': 'LONG',
            'Entry_Price': 100,
            'Qty': 1,
            'SL': 90,
            'Target': 120,
            'Status': 'OPEN',
            'Regime': 'SNIPER',
            'PnL': 0.0,
            'Exit_Price': 0.0,
            'Exit_Reason': '',
            'Exit_Time': ''
        }])
        df.to_csv(self.trade_log, index=False)
        
    def tearDown(self):
        if os.path.exists(self.trade_log):
            os.remove(self.trade_log)

    @patch('live_bot.fetch_data')
    @patch('live_bot.calculate_9ema')
    @patch('live_bot.execute_order')
    def test_hard_sl_hit(self, mock_exec, mock_9ema, mock_fetch):
        # Mock Data: Current Price 85 (Below SL 90)
        mock_intraday = pd.DataFrame({'Close': [100, 95, 85], 'EMA9': [90, 90, 90]})
        mock_fetch.return_value = (pd.DataFrame(), mock_intraday)
        mock_9ema.return_value = mock_intraday
        
        # Run Monitor
        live_bot.monitor_positions()
        
        # Check if trade closed
        df = pd.read_csv(self.trade_log)
        self.assertEqual(df.iloc[0]['Status'], 'CLOSED')
        self.assertIn("Hard SL Hit", df.iloc[0]['Exit_Reason'])
        print("✅ Hard SL Test Passed")

    @patch('live_bot.fetch_data')
    @patch('live_bot.calculate_9ema')
    @patch('live_bot.execute_order')
    def test_strategy_exit(self, mock_exec, mock_9ema, mock_fetch):
        # Reset Log
        self.setUp()
        
        # Mock Data: Price 110 (Above SL), but Closed Candle (105) Below EMA (108)
        # Setup: Last candle (curr) is active. Prev candle (closed) triggers exit.
        
        # Index 0: 100
        # Index 1 (Prev): Close 105, EMA9 108 (Close < EMA -> Exit for LONG)
        # Index 2 (Curr): Close 110
        
        mock_intraday = pd.DataFrame({
            'Close': [100, 105, 110], 
            'EMA9': [90, 108, 108]
        })
        mock_fetch.return_value = (pd.DataFrame(), mock_intraday)
        mock_9ema.return_value = mock_intraday
        
        # Run Monitor
        live_bot.monitor_positions()
        
        # Check if trade closed
        df = pd.read_csv(self.trade_log)
        self.assertEqual(df.iloc[0]['Status'], 'CLOSED')
        self.assertIn("9EMA Reversal", df.iloc[0]['Exit_Reason'])
        print("✅ Strategy Ext Test Passed")

if __name__ == '__main__':
    unittest.main()
