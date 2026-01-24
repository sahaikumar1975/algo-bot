
from day_trading_strategy import fetch_data
import pandas as pd
import numpy as np

daily, df = fetch_data('^NSEI')

# Recalculate ADX as in strategy
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift())
low_close = np.abs(df['Low'] - df['Close'].shift())
ranges = pd.concat([high_low, high_close, low_close], axis=1)
true_range = np.max(ranges, axis=1)
df['ATR'] = true_range.rolling(14).mean()

plus_dm = df['High'] - df['High'].shift(1)
minus_dm = df['Low'].shift(1) - df['Low']
plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

tr_s = true_range.rolling(14).sum()
plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr_s)
minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr_s)
dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
df['ADX'] = dx.rolling(14).mean()

print(df[['Close', 'ADX']].tail(20))
