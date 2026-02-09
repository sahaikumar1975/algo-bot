
import pandas as pd
import unittest
from day_trading_strategy import check_3_candle_setup

class TestStartgySL(unittest.TestCase):
    
    def setUp(self):
        # Create a sample dataframe that forms a valid SHORT setup
        # C1: Open 100, Close 95 (Below 9EMA 98) -> Crossover Sell (Wait: Open > EMA but Close < EMA)
        # C2: Close 94 (Below EMA)
        # C3: Close 93 (Below EMA)
        # Greens in cluster: Maybe C2 was green? Open 93, Close 94.
        
        # Let's construct:
        # EMA9 = 98
        # C1: Open 100, High 102, Low 95, Close 95 (Red)
        # C2: Open 93, High 96, Low 92, Close 94 (Green) -> Ref Candle for Short
        # C3: Open 94, High 95, Low 91, Close 92 (Red)
        # Curr: Close 90 (Break of C2 Low 92)
        
        dates = pd.to_datetime(['2025-01-01 10:00', '2025-01-01 10:05', '2025-01-01 10:10', '2025-01-01 10:15'])
        
        self.df = pd.DataFrame({
            'Open': [100, 93, 94, 94],
            'High': [102, 96, 95, 95], # Cluster High = 102 (C1)
            'Low':  [95, 92, 91, 90],  # C2 Low = 92 (Entry Level)
            'Close':[95, 94, 92, 90],
            'EMA9': [98, 98, 98, 98]
        }, index=dates)
        
    def test_index_sl(self):
        # Index Mode: method='CLUSTER', buffer=5.0
        # Expected SL = Max(Highs) + 5.0 = 102 + 5.0 = 107.0
        
        res = check_3_candle_setup(self.df, sl_method='CLUSTER', limit_buffer=5.0)
        self.assertIsNotNone(res)
        self.assertEqual(res['Signal'], 'SHORT')
        self.assertEqual(res['Entry'], 92) # Low of green candle
        self.assertAlmostEqual(res['SL'], 107.0)
        print("✅ Index SL Logic Passed (Cluster High + 5)")

    def test_stock_sl(self):
        # Stock Mode: method='REF', buffer=0.0
        # Ref Candle is C2 (Green). High is 96.
        # Expected SL = Ref Candle High + 0.0 = 96.0
        
        res = check_3_candle_setup(self.df, sl_method='REF', limit_buffer=0.0)
        self.assertIsNotNone(res)
        self.assertEqual(res['Signal'], 'SHORT')
        self.assertEqual(res['Entry'], 92)
        self.assertAlmostEqual(res['SL'], 96.0)
        print("✅ Stock SL Logic Passed (Ref Candle High)")

if __name__ == '__main__':
    unittest.main()
