"""
Replay Simulation Module
------------------------
Provides ReplaySession class for Streamlit integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from day_trading_strategy import fetch_data, calculate_9ema, check_3_candle_setup, check_9ema_signal

class ReplaySession:
    def __init__(self, ticker="^NSEI", days=2):
        self.ticker = ticker
        self.days = days
        self.data, self.intraday_original = self._load_data()
        self.current_index = 20 # Start with some buffer
        self.balance = 100000.0
        self.position = 0 # 0, 1 (Long), -1 (Short)
        self.entry_price = 0.0
        self.sl = 0.0
        self.target = 0.0
        self.trades = []
        self.pnl_history = []
        self.message_log = []
        
        # Initial Log
        if self.data.empty:
            self.log("No Data Found.")
        else:
            self.log(f"Session Started. {len(self.data)} candles loaded.")

    def _load_data(self):
        try:
            daily, intraday = fetch_data(self.ticker)
            if intraday.empty:
                return pd.DataFrame(), pd.DataFrame()
                
            # Indicators
            intraday = calculate_9ema(intraday)
            
            # Filter last N days
            start_date = datetime.now() - timedelta(days=self.days)
            filter_date = start_date.strftime('%Y-%m-%d')
            intraday = intraday.sort_index()
            intraday = intraday[intraday.index >= filter_date]
            
            return intraday, intraday 
        except Exception as e:
            self.log(f"Error loading data: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def next_candle(self):
        """Advance simulation by one candle."""
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self._check_exit()
            return True
        else:
            self.log("End of Data reached.")
            return False

    def _check_exit(self):
        """Check if current price hits SL/Target."""
        if self.position == 0: return

        curr = self.data.iloc[self.current_index]
        exit_reason = None
        
        if self.position == 1:
            if curr['Low'] <= self.sl: exit_reason = "SL HIT"
            elif curr['High'] >= self.target: exit_reason = "TARGET HIT"
        elif self.position == -1:
            if curr['High'] >= self.sl: exit_reason = "SL HIT"
            elif curr['Low'] <= self.target: exit_reason = "TARGET HIT"
            
        if exit_reason:
            raw_pnl = abs(self.entry_price - (self.sl if 'SL' in exit_reason else self.target))
            if 'SL' in exit_reason: raw_pnl = -raw_pnl
            
            # Index Opt Simulation (High Delta ITM)
            # Nifty: Lot 65, Delta 0.7 -> ~45.5 per point
            # BankNifty: Lot 30, Delta 0.7 -> ~21 per point
            multiplier = 21.0 if 'BANK' in self.ticker else 45.5
            pnl = raw_pnl * multiplier 
            
            self.balance += pnl
            self.trades.append({
                'Time': self.data.index[self.current_index],
                'Type': exit_reason,
                'PnL': pnl,
                'Balance': self.balance
            })
            self.position = 0
            self.log(f"🛑 EXIT: {exit_reason} | PnL: {pnl:.2f}")

    def execute_trade(self, side):
        """Execute a trade at current Close price."""
        if self.position != 0: 
            self.log("⚠️ Application Error: Already in position")
            return
            
        curr = self.data.iloc[self.current_index]
        
        # Check Signal for validity check (optional)
        # Using 3-Candle Check
        # Need last 4 candles
        sig = None
        if self.current_index >= 3:
            df_slice = self.data.iloc[self.current_index-3 : self.current_index+1]
            try:
                sig = check_3_candle_setup(df_slice)
            except Exception as e:
                self.log(f"Strategy Error: {e}")
        
        self.entry_price = curr['Close']
        if sig and sig['Signal'] == side:
            self.sl = sig['SL']
        else:
            # Fallback Manual
            self.sl = curr['Close'] * 0.995 if side=='LONG' else curr['Close']*1.005

        self.position = 1 if side == 'LONG' else -1
        risk = abs(self.entry_price - self.sl)
        self.target = self.entry_price + (risk * 2 * self.position)
        
        self.log(f"🚀 ENTER {side} @ {self.entry_price:.2f} | SL {self.sl:.2f}")

    def get_state(self):
        """Return current state for UI."""
        if self.data.empty or self.current_index >= len(self.data): return None
        
        curr = self.data.iloc[self.current_index]
        prev = self.data.iloc[self.current_index - 1] # Not directly used for 3-candle, we slice
        
        # Check Signal
        sig = None
        is_alert_bull = False
        is_alert_bear = False
        
        if self.current_index >= 3:
            # Pass slice ending at Current
            df_slice = self.data.iloc[self.current_index-3 : self.current_index+1]
            try:
                sig = check_3_candle_setup(df_slice)
            except: pass
            
            # Alert Visuals (Logic is complex now, simplified for UI)
            # Check if last 3 closed allow setup (potential alert)
            c1 = self.data.iloc[self.current_index-3]
            c2 = self.data.iloc[self.current_index-2]
            c3 = self.data.iloc[self.current_index-1] # Last Closed
             
            if c1['Close'] > c1['EMA9'] and c2['Close'] > c2['EMA9'] and c3['Close'] > c3['EMA9']:
                 is_alert_bull = True # Potential Long setup forming
            elif c1['Close'] < c1['EMA9'] and c2['Close'] < c2['EMA9'] and c3['Close'] < c3['EMA9']:
                 is_alert_bear = True

        open_pnl = 0
        if self.position != 0:
            diff = (curr['Close'] - self.entry_price) if self.position == 1 else (self.entry_price - curr['Close'])
            multiplier = 21.0 if 'BANK' in self.ticker else 45.5
            open_pnl = diff * multiplier
        
        return {
            'timestamp': self.data.index[self.current_index],
            'price': curr['Close'],
            'ema9': curr['EMA9'],
            'open_pnl': open_pnl,
            'signal': sig,
            'alert_bull': is_alert_bull, # Now means "Cluster Formed"
            'alert_bear': is_alert_bear,
            'balance': self.balance,
            'position': 'LONG' if self.position == 1 else 'SHORT' if self.position == -1 else 'FLAT'
        }

    def log(self, msg):
        timestamp = datetime.now()
        if not self.data.empty and self.current_index < len(self.data):
             timestamp = self.data.index[self.current_index]
             
        self.message_log.insert(0, f"[{timestamp}] {msg}")
