"""Chart generation module for creating candlestick charts with SMA overlays."""

import io
import os
import tempfile
from typing import Optional
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt


def generate_chart(
    df: pd.DataFrame,
    ticker: str,
    sma21: Optional[pd.Series] = None,
    sma50: Optional[pd.Series] = None,
    sma200: Optional[pd.Series] = None,
    last_n_days: int = 60,
    save_path: Optional[str] = None
) -> bytes:
    """Generate a candlestick chart with SMA overlays.

    Args:
        df: DataFrame with OHLC data (columns: Open, High, Low, Close, Volume)
        ticker: Stock ticker symbol for the title
        sma21: Series with SMA21 values (optional, will calculate if not provided)
        sma50: Series with SMA50 values (optional, will calculate if not provided)
        sma200: Series with SMA200 values (optional, will calculate if not provided)
        last_n_days: Number of days to display on chart (default 60)
        save_path: Optional path to save the chart image

    Returns:
        bytes: PNG image data
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Ensure we have proper OHLC columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Make a copy and calculate SMAs if not provided
    chart_df = df.copy()

    if sma21 is None:
        chart_df['SMA21'] = chart_df['Close'].rolling(window=21, min_periods=1).mean()
    else:
        chart_df['SMA21'] = sma21

    if sma50 is None:
        chart_df['SMA50'] = chart_df['Close'].rolling(window=50, min_periods=1).mean()
    else:
        chart_df['SMA50'] = sma50

    if sma200 is None:
        chart_df['SMA200'] = chart_df['Close'].rolling(window=200, min_periods=1).mean()
    else:
        chart_df['SMA200'] = sma200

    # Limit to last N days for better visibility
    if len(chart_df) > last_n_days:
        chart_df = chart_df.tail(last_n_days)

    # Ensure index is DatetimeIndex for mplfinance
    if not isinstance(chart_df.index, pd.DatetimeIndex):
        chart_df.index = pd.to_datetime(chart_df.index)

    # Create additional plots for SMAs
    add_plots = [
        mpf.make_addplot(chart_df['SMA21'], color='#2196F3', width=1.5, label='SMA21'),
        mpf.make_addplot(chart_df['SMA50'], color='#FF9800', width=1.5, label='SMA50'),
        mpf.make_addplot(chart_df['SMA200'], color='#9C27B0', width=1.5, label='SMA200'),
    ]

    # Custom style
    mc = mpf.make_marketcolors(
        up='#26A69A',
        down='#EF5350',
        edge='inherit',
        wick='inherit',
        volume='in'
    )
    style = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='-',
        gridcolor='#E0E0E0',
        facecolor='white',
        figcolor='white'
    )

    # Create figure
    fig, axes = mpf.plot(
        chart_df,
        type='candle',
        style=style,
        title=f'{ticker} - SMA Crossover Signal',
        ylabel='Price',
        volume=True if 'Volume' in chart_df.columns else False,
        addplot=add_plots,
        figsize=(12, 8),
        returnfig=True,
        tight_layout=True
    )

    # Add legend manually
    ax = axes[0]
    ax.legend(['SMA21', 'SMA50', 'SMA200'], loc='upper left', fontsize=9)

    # Add signal annotation (arrow pointing to last candle)
    last_date = chart_df.index[-1]
    last_close = chart_df['Close'].iloc[-1]
    ax.annotate(
        'Signal',
        xy=(len(chart_df) - 1, last_close),
        xytext=(len(chart_df) - 5, last_close * 1.03),
        fontsize=10,
        color='green',
        arrowprops=dict(arrowstyle='->', color='green', lw=1.5)
    )

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    image_bytes = buf.read()
    buf.close()
    plt.close(fig)

    # Optionally save to file
    if save_path:
        with open(save_path, 'wb') as f:
            f.write(image_bytes)

    return image_bytes


def generate_chart_for_result(
    ticker: str,
    ohlc_df: pd.DataFrame,
    output_dir: Optional[str] = None,
    last_n_days: int = 60
) -> tuple:
    """Generate a chart for a screening result.

    Args:
        ticker: Stock ticker symbol
        ohlc_df: DataFrame with OHLC data and SMA columns
        output_dir: Directory to save chart (uses temp dir if not specified)
        last_n_days: Number of days to display

    Returns:
        tuple: (image_bytes, file_path)
    """
    if output_dir is None:
        output_dir = tempfile.gettempdir()

    # Clean ticker name for filename
    clean_ticker = ticker.replace('.', '_').replace(':', '_')
    file_path = os.path.join(output_dir, f'{clean_ticker}_chart.png')

    # Get SMA series if available
    sma21 = ohlc_df.get('SMA21')
    sma50 = ohlc_df.get('SMA50')
    sma200 = ohlc_df.get('SMA200')

    image_bytes = generate_chart(
        df=ohlc_df,
        ticker=ticker,
        sma21=sma21,
        sma50=sma50,
        sma200=sma200,
        last_n_days=last_n_days,
        save_path=file_path
    )

    return image_bytes, file_path


if __name__ == '__main__':
    # Test with sample data
    import yfinance as yf

    ticker = 'RELIANCE.NS'
    tk = yf.Ticker(ticker)
    df = tk.history(period='180d', interval='1d')

    if not df.empty:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        image_bytes, path = generate_chart_for_result(ticker, df)
        print(f'Chart saved to: {path}')
        print(f'Image size: {len(image_bytes)} bytes')
