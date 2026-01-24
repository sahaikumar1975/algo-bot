"""Fetch NIFTY 500 constituents (symbols) from Wikipedia and write to `stocks/nifty500.csv`.

This script uses `pandas.read_html` to parse the constituents table on the NIFTY 500 Wikipedia page.
Run this once to generate a local `stocks/nifty500.csv` in Yahoo format (adds `.NS`).
"""
import pandas as pd
import re
from pathlib import Path

WIKI_URL = 'https://en.wikipedia.org/wiki/NIFTY_500'


def normalize_symbol(sym: str) -> str:
    # remove whitespace and ensure uppercase
    s = sym.strip().upper()
    # some symbols contain characters like '&' or '.' keep as-is
    # append .NS for Yahoo India format unless already present
    if not s.endswith('.NS'):
        s = s + '.NS'
    return s


def fetch_and_write(path: Path = Path('stocks/nifty500.csv')):
    print('Fetching NIFTY 500 constituents from Wikipedia...')
    tables = pd.read_html(WIKI_URL)
    # find a table containing 'Symbol' or 'Company Name'
    sym_col = None
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str)]
        if any('symbol' in c for c in cols) or any('company name' in c for c in cols):
            # try to identify symbol column
            for c in t.columns:
                if 'symbol' in str(c).lower():
                    sym_col = str(c)
                    break
        if sym_col:
            symbols = t[sym_col].astype(str).tolist()
            break
    if not sym_col:
        raise RuntimeError('Could not find constituents table on the page. Try running later or update the parser.')

    # normalize and write
    cleaned = []
    for s in symbols:
        s = re.sub(r"\s+", "", s)
        if not s or s.lower().startswith('sl.no'):
            continue
        cleaned.append(normalize_symbol(s))

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for s in cleaned:
            f.write(s + '\n')
    print(f'Wrote {len(cleaned)} symbols to {path}')


if __name__ == '__main__':
    fetch_and_write()
