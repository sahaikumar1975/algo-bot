"""Server-like scheduler: periodically screens NIFTY500, sends Telegram notifications,
and optionally places orders via Fyers.

Run this on a server/VM or as a background service. It reads `config.json` in the
project root (copy `config_example.json` → `config.json` and fill values).
"""
import json
import time
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from strategy import screen_tickers
from telegram import send_telegram_message
from trading import compute_quantity_from_risk
from fyers_integration import FyersClient
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

CONFIG_PATH = Path('config.json')
TICKER_PATH = Path('stocks/nifty500.csv')


def load_config():
    if not CONFIG_PATH.exists():
        raise RuntimeError('Missing config.json — copy config_example.json to config.json and edit')
    return json.loads(CONFIG_PATH.read_text())


def load_tickers():
    if not TICKER_PATH.exists():
        # attempt to build using fetch_nifty500
        from fetch_nifty500 import fetch_and_write
        fetch_and_write(TICKER_PATH)
    return [l.strip() for l in TICKER_PATH.read_text().splitlines() if l.strip()]


def job(config):
    logging.info('Starting scheduled screener job')
    tickers = load_tickers()

    # Check if charts are enabled in config
    send_charts = config.get('send_charts', True)
    vol_multiplier = config.get('vol_multiplier', 1.5)

    results, signal_type = screen_all(tickers, period='180d', include_df=send_charts,
                                      vol_multiplier=vol_multiplier)
    if not results:
        logging.info('No matches found on this run (neither crossover nor volume breakout)')
        return

    logging.info(f'Found {len(results)} stocks closing above SMA21')

    # Build message
    text_lines = [f'<b>Close Above SMA21 — {len(results)} stocks</b>']
    text_lines.append('(Price above SMA50 & SMA200)')
    text_lines.append('')  # Empty line

    for r in results:
        line = f"📊 {r['ticker']} — close {r['close']:.2f} | SMA21 {r['sma21']:.2f}"
        text_lines.append(line)
    message = '\n'.join(text_lines)

    # send to telegram if configured
    tok = config.get('telegram_token')
    chat = config.get('telegram_chat')
    if tok and chat:
        # Send text summary first
        resp = send_telegram_message(tok, chat, message)
        if resp:
            logging.info('Sent text results to Telegram')
        else:
            logging.warning('Failed to send text to Telegram')

        # Send charts if enabled
        if send_charts:
            logging.info('Generating and sending charts...')
            for r in results:
                try:
                    if 'ohlc_df' not in r:
                        logging.warning(f"No OHLC data for {r['ticker']}, skipping chart")
                        continue

                    image_bytes, _ = generate_chart_for_result(
                        ticker=r['ticker'],
                        ohlc_df=r['ohlc_df'],
                        last_n_days=60
                    )
                    caption = f"<b>{r['ticker']}</b>\nClose: {r['close']:.2f} | SMA21: {r['sma21']:.2f} | SMA50: {r['sma50']:.2f}"
                    resp = send_telegram_photo(tok, chat, image_bytes, caption)
                    if resp:
                        logging.info(f"Sent chart for {r['ticker']}")
                    else:
                        logging.warning(f"Failed to send chart for {r['ticker']}")
                except Exception as e:
                    logging.exception(f"Error generating/sending chart for {r['ticker']}: {e}")

    # optional trading via fyers
    fyers_cfg = config.get('fyers', {})
    trade_cfg = config.get('trade', {})
    if fyers_cfg.get('enabled') and trade_cfg.get('enabled'):
        client = FyersClient(access_token=fyers_cfg.get('access_token'), base_url=fyers_cfg.get('base_url', 'https://api.fyers.in'))
        capital = float(trade_cfg.get('capital', 100000))
        risk = float(trade_cfg.get('risk_per_trade', 0.01))
        sl_pct = float(trade_cfg.get('initial_sl_pct', 0.02))
        for r in results:
            price = float(r['close'])
            stop_price = price * (1 - sl_pct)
            qty = compute_quantity_from_risk(capital, price, risk, stop_price)
            if qty <= 0:
                logging.info(f"Skipping {r['ticker']}: qty computed 0")
                continue
            try:
                # NOTE: adapt `symbol` format to match Fyers expectation
                resp = client.place_order(symbol=r['ticker'].replace('.NS',''), qty=qty, side='BUY', order_type='MARKET')
                logging.info(f"Placed order for {r['ticker']}: qty={qty} resp={resp}")
            except Exception as e:
                logging.exception(f"Order failed for {r['ticker']}: {e}")


def main():
    config = load_config()
    sched = BackgroundScheduler()

    # schedule using cron values from config
    cron = config.get('schedule', {})
    hour = cron.get('cron_hour', 15)
    minute = cron.get('cron_minute', 31)
    days = cron.get('days_of_week', 'mon-fri')

    sched.add_job(job, 'cron', day_of_week=days, hour=hour, minute=minute, args=[config], id='screener')
    sched.start()
    logging.info(f'Scheduler started — job scheduled at {hour:02d}:{minute:02d} {days}')

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        logging.info('Shutting down scheduler')
        sched.shutdown()


if __name__ == '__main__':
    main()
