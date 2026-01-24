# SMA 21/50 Crossover Screener (above SMA200) — Nifty 500

This project provides a small web app that screens Nifty 500 stocks for a completed SMA21/SMA50 crossover where both SMAs are above SMA200. It can list results in the UI and send them to a Telegram channel. A template for automated order execution (broker integration) is included.

Not production ready — broker integration requires API credentials and careful testing before live trading.

Quick start (local):

1. Create a Python virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

Files of interest:

- `app.py`: Streamlit web UI — upload tickers / use sample, run screener, send to Telegram.
- `strategy.py`: SMA calculations and signal detection.
- `telegram.py`: Helper to send messages to a Telegram bot/channel.
- `trading.py`: Template functions for position sizing, SL, trailing SL and broker hooks.
- `stocks/nifty500.csv`: Sample Nifty tickers (you should replace with full list).

Additional scripts:

- `fetch_nifty500.py`: Fetches the full NIFTY 500 constituents from Wikipedia and writes `stocks/nifty500.csv` (Yahoo format `.NS`).
- `fyers_integration.py`: Minimal Fyers API wrapper (placeholders — adapt to Fyers docs and test in paper mode).
- `server.py`: Scheduler that runs the screener on a cron schedule, posts results to Telegram, and can optionally place orders via Fyers.
- `config_example.json`: Example configuration for `server.py` (copy to `config.json` and fill values).

Telegram setup:

- Create a bot via BotFather and get the token.
- Add the bot to your channel as an admin to post messages, or use a chat ID for a group.

Important notes and next steps:

- Validate ticker formats for your data source (Yahoo uses `.NS` suffix for NSE tickers, e.g. `RELIANCE.NS`).
- Backtest strategy on historical data and paper trade before using real capital.
- For automation, integrate with a broker SDK (Zerodha Kite Connect / Upstox / Interactive Brokers) and add safeguards: order throttling, retries, logging, and circuit-breakers.

Server (scheduled runs)

- Copy `config_example.json` to `config.json` and edit with your Telegram bot token, chat id, and (optionally) Fyers access token.
- Ensure `stocks/nifty500.csv` exists. To build it automatically, run:

```bash
python fetch_nifty500.py
```

- Start the scheduler (recommended on a server or VM):

```bash
python server.py
```

The scheduler will run at the time configured in `config.json` (default shown in `config_example.json`) and will:
- Run the SMA screener across the `stocks/nifty500.csv` list
- Post results to Telegram if configured
- Optionally place orders via Fyers when `trade.enabled` and `fyers.enabled` are true (adapt `fyers_integration.py` for exact payloads and test in paper mode)

Fyers integration notes

- `fyers_integration.py` offers a minimal `FyersClient` that posts to `POST /api/v2/orders` with a simple payload. Fyers' API payloads and symbol formats can differ (e.g., `NSE:RELIANCE-EQ`); update the wrapper to match the exact API documentation and test thoroughly in paper mode before placing real orders.

Security & production

- Do not commit real API keys to the repository. Use environment variables or a secure secrets manager for production deployments.
- Add logging, monitoring, circuit breakers, and manual confirmation steps before enabling live order placements.


If you want, I can:
- Add a full Nifty 500 ticker list file.
- Add a desktop wrapper (Electron/pyinstaller) so the app can be launched without terminal.
- Integrate with a broker API (you will need to provide API keys and grant test access).
