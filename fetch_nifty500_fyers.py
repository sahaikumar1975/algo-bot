"""Fetch NIFTY 500 constituents from Wikipedia and write to `stocks/nifty500.csv`
in Fyers format: NSE:<SYMBOL>-EQ (one per line).

Run this once to generate the ticker list for the screener and server.
"""
import pandas as pd
from pathlib import Path

WIKI_URL = 'https://en.wikipedia.org/wiki/NIFTY_500'


def fyers_format(symbol: str) -> str:
    """Convert raw symbol to Fyers format: NSE:<SYMBOL>-EQ"""
    s = symbol.strip().upper()
    # handle special cases with & or .
    s = s.replace('&', '&')  # keep as-is
    return f'NSE:{s}-EQ'


def fetch_and_write(path: Path = Path('stocks/nifty500.csv')):
    """Fetch NIFTY 500 constituents from Wikipedia and write in Fyers format."""
    print('Fetching NIFTY 500 constituents from Wikipedia...')
    try:
        tables = pd.read_html(WIKI_URL)
    except Exception as e:
        print(f'Error fetching page: {e}')
        return

    symbols = []
    for tbl in tables:
        cols_lower = [str(c).lower() for c in tbl.columns]
        if any('symbol' in c for c in cols_lower):
            # found the constituents table
            for col in tbl.columns:
                if 'symbol' in str(col).lower():
                    raw_syms = tbl[col].astype(str).tolist()
                    for s in raw_syms:
                        s = s.strip()
                        if s and s.upper() not in ['SYMBOL', 'NAN', '']:
                            symbols.append(s)
                    break
            if symbols:
                break

    if not symbols:
        print('No symbols found. Check the page structure.')
        return

    # dedupe and format
    symbols = list(dict.fromkeys(symbols))  # remove duplicates while preserving order
    fyers_symbols = [fyers_format(s) for s in symbols]

    # write
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for sym in fyers_symbols:
            f.write(sym + '\n')

    print(f'✓ Wrote {len(fyers_symbols)} symbols to {path}')
    print('Sample tickers:')
    for sym in fyers_symbols[:5]:
        print(f'  {sym}')


if __name__ == '__main__':
    fetch_and_write()
