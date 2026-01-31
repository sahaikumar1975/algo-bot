"""
Professional Swing Trading Dashboard

Features:
- Stock screener with multi-indicator analysis
- Individual stock analysis with all technical indicators
- Strategy backtesting with performance metrics
- Interactive charts with indicator overlays
- Risk calculator for position sizing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import re
from dotenv import load_dotenv
import yfinance as yf
from fyers_integration import FyersApp
import json

# Load environment variables
load_dotenv()

CONFIG_FILE = "bot_config.json"

def load_config():
    """Load bot configuration."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "MAX_DAILY_TRADES": 9,
        "MAX_STOCK_TRADES": 5,
        "MAX_NIFTY_TRADES": 2,
        "MAX_BANKNIFTY_TRADES": 2,
        "CAPITAL": 100000,
        "ALLOCATION_PER_TRADE": 10000,
        "RISK_PER_TRADE_PERCENT": 1.0,
        "MAX_DAILY_LOSS_PERCENT": 2.0
    }

def save_config(config):
    """Save bot configuration."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


def set_key(file_path, key, value):
    """Updates or adds a key-value pair in the .env file."""
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write(f"{key}={value}\n")
        return

    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    key_found = False
    
    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}\n")
            key_found = True
        else:
            new_lines.append(line)
    
    if not key_found:
        if new_lines and not new_lines[-1].endswith('\n'):
            new_lines[-1] += '\n'
        new_lines.append(f"{key}={value}\n")

    with open(file_path, "w") as f:
        f.writelines(new_lines)

from swing_strategy import (
    calculate_sector_performance, get_sector_momentum_score, 
    get_swing_focus_sectors, get_top_stocks_by_sector
)

# Import Day Trading Modules
# Import Day Trading Modules
from day_trading_strategy import backtest_day_strategy
from market_scanner import run_scanner, NIFTY_50
from fyers_integration import FyersApp  # Added Fyers Integration
import subprocess
import time

# Page configuration
st.set_page_config(
    page_title="Nifty 200 Swing Trading App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State for Fyers
if 'fyers_token' not in st.session_state:
    st.session_state['fyers_token'] = None

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .signal-grade-A { color: #00ff00; font-weight: bold; font-size: 24px; }
    .signal-grade-B { color: #90EE90; font-weight: bold; font-size: 24px; }
    .signal-grade-C { color: #ffff00; font-weight: bold; font-size: 24px; }
    .signal-grade-D { color: #ff6b6b; font-weight: bold; font-size: 24px; }
    .positive { color: #00ff00; }
    .negative { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)

# --------------------------
# SIDEBAR: Fyers Login
# --------------------------
with st.sidebar.expander("🔐 Fyers Live Login", expanded=False):
    # Try to load from Secrets, else Empty
    # Hardcoded Credentials
    # Try to load from Secrets, else Empty
    # Hardcoded Credentials (Fallback)
    default_app_id = os.getenv("FYERS_APP_ID", "SW02VFH9PP-100")
    default_secret = os.getenv("FYERS_SECRET", "2XQJ9KWTLR")
    
    fyers_app_id = st.text_input("Client ID (App ID)", value=default_app_id, help="e.g., XCXXXXX-100", key="fyers_app_id_new")
    fyers_secret = st.text_input("Secret Key", value=default_secret, type="password", key="fyers_secret_new")
    fyers_redirect = st.text_input("Redirect URI", value="https://trade.fyers.in/api-login/redirect-uri/index.html")

    if st.button("Generate Login Link"):
        fyers_app_id = fyers_app_id.strip()
        fyers_secret = fyers_secret.strip()
        fyers_redirect = fyers_redirect.strip()
        
        if fyers_app_id and fyers_secret:
            # Save to .env for persistence
            set_key(".env", "FYERS_APP_ID", fyers_app_id)
            set_key(".env", "FYERS_SECRET", fyers_secret)
            
            app = FyersApp(fyers_app_id, fyers_secret, redirect_uri=fyers_redirect)
            auth_url = app.get_login_url()
            st.markdown(f"[**👉 Click to Login**]({auth_url})", unsafe_allow_html=True)
            st.info("1. Click Link & Login\n2. Copy 'auth_code' from URL\n3. Paste below")
            st.success("Credentials saved to .env")
        else:
            st.error("Enter App ID & Secret")

    auth_code_input = st.text_input("Paste Auth Code Here (or full 404 URL)")
    
    if st.button("Authenticate & Connect"):
        fyers_app_id = fyers_app_id.strip()
        fyers_secret = fyers_secret.strip()
        
        # Smart Extraction: Handle full URL paste
        # Smart Extraction: Handle full URL paste
        if "http" in auth_code_input:
                try:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(auth_code_input)
                    params = urllib.parse.parse_qs(parsed.query)
                    
                    if 'error_msg' in params:
                        st.error(f"❌ Login Error from Fyers: {params['error_msg'][0]}")
                        st.warning("💡 Hint: checks your App ID and Secret. 'invalid appId' means the Client ID is wrong.")
                        st.stop()
                    
                    if 'auth_code' in params:
                        auth_code_input = params['auth_code'][0]
                        st.success(f"✅ Extracted Code: {auth_code_input[:10]}...{auth_code_input[-5:]}")
                    else:
                         st.error("❌ URL detected, but 'auth_code' not found. Ensure you copied the FULL redirected URL.")
                         st.stop()
                except Exception as e:
                    st.error(f"❌ Error parsing URL: {e}")
                    st.stop()
        
        auth_code_input = auth_code_input.strip()

        if fyers_app_id and fyers_secret and auth_code_input:
            try:
                app = FyersApp(fyers_app_id, fyers_secret, redirect_uri=fyers_redirect)
                token = app.generate_access_token(auth_code_input)
                st.session_state['fyers_token'] = token
                st.session_state['fyers_client_id'] = fyers_app_id 
                st.success("✅ Connected! Token Generated.")
                st.rerun() # Refresh to update sidebar state
            except Exception as e:
                masked_code = f"{auth_code_input[:5]}...{auth_code_input[-5:]}" if len(auth_code_input) > 10 else "SHORT_CODE"
                st.error(f"Login Failed: {e} | Code Used: {masked_code}")
                st.warning("⚠️ Auth Codes expire in < 1 minute and are ONE-TIME use. Generate a NEW Link!")
        else:
            st.error("Missing Credentials or Auth Code")

    if st.session_state['fyers_token']:
        st.caption("🟢 Live Broker Connected")
        enable_live = st.checkbox("Enable Live Execution", value=False, help="If checked, Bot will place REAL ORDERS.")
        if enable_live:
            st.session_state['live_trading_active'] = True
            st.warning("⚠️ REAL MONEY MODE ON")
        else:
            st.session_state['live_trading_active'] = False

st.sidebar.divider()


def load_nifty200():
    """Load Nifty 200 stock list."""
    csv_path = os.path.join(os.path.dirname(__file__), 'stocks', 'nifty200.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'Symbol' in df.columns:
            return df['Symbol'].tolist()
    return []


def create_candlestick_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Create an interactive candlestick chart with all indicators."""

    # Create subplots: 4 rows
    # Row 1: Candlestick with SMAs and Bollinger Bands (60%)
    # Row 2: Volume (10%)
    # Row 3: RSI (15%)
    # Row 4: MACD (15%)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=(f'{ticker} - Price & Indicators', 'Volume', 'RSI', 'MACD')
    )

    # Row 1: Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00ff00',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )

    # SMAs
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA21'], name='SMA21',
                   line=dict(color='#FFD700', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA50'], name='SMA50',
                   line=dict(color='#00BFFF', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['SMA200'], name='SMA200',
                   line=dict(color='#FF69B4', width=1.5)),
        row=1, col=1
    )

    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                   line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dot')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                   line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dot'),
                   fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
        row=1, col=1
    )

    # Row 2: Volume
    colors = ['#00ff00' if df['Close'].iloc[i] >= df['Open'].iloc[i]
              else '#ff4444' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Vol_SMA20'], name='Vol SMA20',
                   line=dict(color='orange', width=1)),
        row=2, col=1
    )

    # Row 3: RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                   line=dict(color='#9370DB', width=1.5)),
        row=3, col=1
    )
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)

    # Row 4: MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                   line=dict(color='#00BFFF', width=1.5)),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                   line=dict(color='#FF6347', width=1.5)),
        row=4, col=1
    )
    # MACD Histogram
    colors_macd = ['#00ff00' if val >= 0 else '#ff4444' for val in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram',
               marker_color=colors_macd, opacity=0.5),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        height=900,
        template='plotly_dark',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=4, col=1)

    return fig


def create_equity_curve(equity_data: list, trades: list, ticker: str) -> go.Figure:
    """Create equity curve chart with trade markers."""
    df = pd.DataFrame(equity_data)

    fig = go.Figure()

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#00BFFF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,191,255,0.1)'
        )
    )

    # Add trade markers
    for trade in trades:
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_date']],
                y=[df[df['date'] == trade['entry_date']]['equity'].values[0] if len(df[df['date'] == trade['entry_date']]) > 0 else None],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Entry',
                showlegend=False
            )
        )
        # Exit marker
        color = 'green' if trade['pnl'] > 0 else 'red'
        fig.add_trace(
            go.Scatter(
                x=[trade['exit_date']],
                y=[df[df['date'] == trade['exit_date']]['equity'].values[0] if len(df[df['date'] == trade['exit_date']]) > 0 else None],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color=color),
                name='Exit',
                showlegend=False
            )
        )

    fig.update_layout(
        title=f'Equity Curve - {ticker}',
        template='plotly_dark',
        height=400,
        xaxis_title='Date',
        yaxis_title='Equity (₹)',
        showlegend=True
    )

    return fig


def display_signal_strength(strength_data: dict):
    """Display signal strength breakdown."""
    total = strength_data['total_score']
    grade = strength_data['grade']
    breakdown = strength_data['breakdown']

    # Grade color
    if grade.startswith('A'):
        grade_class = 'signal-grade-A'
    elif grade.startswith('B'):
        grade_class = 'signal-grade-B'
    elif grade.startswith('C'):
        grade_class = 'signal-grade-C'
    else:
        grade_class = 'signal-grade-D'

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"### Signal Grade")
        st.markdown(f'<span class="{grade_class}">{grade}</span> ({total}/100)', unsafe_allow_html=True)

    with col2:
        st.markdown("### Breakdown")
        metrics = {
            'Trend Alignment': (breakdown.get('trend', 0), 25),
            'RSI Position': (breakdown.get('rsi', 0), 20),
            'MACD Momentum': (breakdown.get('macd', 0), 20),
            'ADX Strength': (breakdown.get('adx', 0), 15),
            'Volume': (breakdown.get('volume', 0), 10),
            'Bollinger': (breakdown.get('bollinger', 0), 10)
        }

        for name, (score, max_score) in metrics.items():
            pct = score / max_score * 100
            color = '#00ff00' if pct >= 70 else '#ffff00' if pct >= 40 else '#ff6b6b'
            st.markdown(f"**{name}**: {score}/{max_score}")
            st.progress(pct / 100)


# --- DAY TRADING CHART FUNCTION ---
def plot_day_trading_chart(df, trades, ticker):
    """
    Plots interactive candlestick chart with CPR and Trade Signals.
    """
    fig = go.Figure()

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='Price'
    ))

    # 2. CPR Lines (TC, Pivot, BC)
    fig.add_trace(go.Scatter(x=df.index, y=df['TC'], mode='lines', name='TC', line=dict(color='blue', width=1, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Pivot'], mode='lines', name='Pivot', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BC'], mode='lines', name='BC', line=dict(color='blue', width=1, dash='dot')))
    
    # 3. VWAP
    if 'VWAP' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='orange', width=1)))

    # 4. Trade Markers
    if trades:
        for t in trades:
            if 'Time' in t:
                # Entry Marker
                side = t.get('Side', 'Long')
                entry_color = 'green' if side == 'Long' else 'red'
                entry_symbol = 'triangle-up' if side == 'Long' else 'triangle-down'
                
                fig.add_trace(go.Scatter(
                    x=[t['Time']], y=[t['Entry']],
                    mode='markers',
                    marker=dict(symbol=entry_symbol, size=12, color=entry_color),
                    name=f'{side} Entry', showlegend=False
                ))
                # Exit Marker
                color = 'red' if t['Result'] == 'Loss' else 'green'
                symbol = 'x' if t['Result'] == 'Loss' else 'triangle-down'
                fig.add_trace(go.Scatter(
                    x=[t['Time']], 
                    y=[t['Exit']],
                    mode='markers',
                    marker=dict(symbol=symbol, size=12, color=color),
                    name='Exit', showlegend=False
                ))

    fig.update_layout(
        title=f'{ticker} - Day Trading Analysis',
        yaxis_title='Price',
        xaxis_title='Time',
        height=600,
        template='plotly_dark'
    )
    return fig


def main():
    st.title("🚀 SMA2150 Command Center")
    st.markdown("*Autonomous Day Trading Bot & Sector Analytics*")
    st.markdown("---")
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Unified Navigation
    page = st.sidebar.radio(
        "Modules",
        [
            "🤖 Live Bot Manager", 
            "📜 Paper Trading P&L", 
            "📡 Intraday Scanner", 
            "🏭 Sector Analysis",
            "⚡ Backtest Day Strategy", 
            "⚡ Backtest Indices (Options)"
        ],
        key="nav_radio"
    )

    # Load stock list
    nifty200 = load_nifty200()

    # Initialize session states
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []
    if 'trade_notes' not in st.session_state:
        st.session_state.trade_notes = {}
    if 'screener_results' not in st.session_state:
        st.session_state.screener_results = []
    if 'selected_stock_for_analysis' not in st.session_state:
        st.session_state.selected_stock_for_analysis = None
        
    # --- DAY TRADING PAGES ---
    if page == "🤖 Live Bot Manager":
        st.header("🤖 Day Trading Bot Manager")
        st.markdown("*Control the autonomous trading bot and monitor live logs.*")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        bot_running = False
        # Check if process is running
        try:
            # Simple check for python process with 'live_bot.py'
            # Robust check using pgrep
            cmd = "pgrep -f live_bot.py"
            output = subprocess.check_output(cmd, shell=True).decode()
            if output.strip():
                bot_running = True
        except subprocess.CalledProcessError:
            bot_running = False
        except Exception as e:
            bot_running = False
            
        # AI Configuration
        with st.expander("🤖 AI Configuration (Gemini)", expanded=False):
            # Load from env or session state
            saved_key = os.getenv("GEMINI_API_KEY")
            
            # Mask the key for display if it exists
            display_value = saved_key if saved_key else ""
            
            api_key = st.text_input("Gemini API Key", value=display_value, type="password", key="gemini_key", help="Enter your new key from Google AI Studio")
            
            if st.button("Save API Key"):
                if api_key:
                    # Save to .env file safely
                    set_key(".env", "GEMINI_API_KEY", api_key)
                    
                    # Update Session & Env in Runtime
                    os.environ["GEMINI_API_KEY"] = api_key
                    st.session_state['gemini_api_key'] = api_key
                    st.success("✅ Key saved securely to .env file!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Please enter a valid key.")

            if saved_key:
                st.caption("✅ Key is loaded from .env")
            else:
                st.warning("⚠️ No API Key found. AI features will be disabled.")

                st.warning("⚠️ No API Key found. AI features will be disabled.")

        with col1:
            st.metric("Bot Status", "RUNNING 🟢" if bot_running else "STOPPED 🔴")
    
        with col2:
            if bot_running:
                if st.button("🛑 STOP BOT", type="primary"):
                    os.system("pkill -f live_bot.py")
                    st.success("Bot stopped.")
                    time.sleep(1)
                    st.rerun()
            else:
                if st.button("▶️ START BOT", type="primary"):
                    # Prepare Env
                    env = os.environ.copy()
                    if 'gemini_api_key' in st.session_state:
                        env['GEMINI_API_KEY'] = st.session_state['gemini_api_key']
                    
                    # Only enable Live Trade if Toggle is ON and Token exists
                    if st.session_state.get('live_trading_active', False) and st.session_state.get('fyers_token'):
                        env['FYERS_TOKEN'] = st.session_state['fyers_token']
                        env['FYERS_CLIENT_ID'] = st.session_state.get('fyers_client_id', '')
                        st.warning("⚠️ Bot Starting in LIVE TRADING MODE")
                    else:
                        st.info("Bot Starting in PAPER TRADING MODE")

                    # Start in background
                    subprocess.Popen(["python3", "live_bot.py"], env=env)
                    st.success("Bot started.")
                    time.sleep(1)
                    st.rerun()
        
                    st.rerun()
                    
        # --- MARKET TREND DASHBOARD ---
        st.subheader("📊 Market Condition (Indices)")
        if os.path.exists("daily_scan_results.csv"):
            try:
                res_df = pd.read_csv("daily_scan_results.csv")
                
                # Extract Index Trends
                nifty = res_df[res_df['Ticker'] == '^NSEI']
                banknifty = res_df[res_df['Ticker'] == '^NSEBANK']
                
                m1, m2 = st.columns(2)
                
                with m1:
                    trend = "NEUTRAL"
                    if not nifty.empty: trend = nifty.iloc[0]['Trend']
                    color = "off"
                    if trend == "BULLISH": color = "normal" 
                    elif trend == "BEARISH": color = "inverse"
                    st.metric("NIFTY 50", trend, delta="Up" if trend=="BULLISH" else "Down" if trend=="BEARISH" else None, delta_color=color)
                    
                with m2:
                    trend = "NEUTRAL"
                    if not banknifty.empty: trend = banknifty.iloc[0]['Trend']
                    color = "off"
                    if trend == "BULLISH": color = "normal"
                    elif trend == "BEARISH": color = "inverse"
                    st.metric("BANK NIFTY", trend, delta="Up" if trend=="BULLISH" else "Down" if trend=="BEARISH" else None, delta_color=color)
            except Exception as e:
                st.error(f"Error loading trends: {e}")
        else:
            st.info("Market Scan not yet run today.")

        # --- Risk Management Section ---
        with st.expander("💰 Risk & Fund Management", expanded=False):
            config = load_config()
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 💸 Capital Allocation")
                new_capital = st.number_input("Total Capital (₹)", value=config.get("CAPITAL", 100000), step=5000)
                new_allocation = st.number_input("Allocation Per Trade (₹)", value=config.get("ALLOCATION_PER_TRADE", 10000), step=1000, help="Amount to invest in each stock trade.")
                
                st.markdown("##### 🛡️ Risk Limits")
                new_risk_pct = st.number_input("Risk Per Trade (%)", value=float(config.get("RISK_PER_TRADE_PERCENT", 1.0)), step=0.1)
                new_max_loss = st.number_input("Max Daily Loss (%)", value=float(config.get("MAX_DAILY_LOSS_PERCENT", 2.0)), step=0.5)

            with c2:
                st.markdown("##### 🔢 Trade Frequency Limits")
                new_stock_limit = st.number_input("Max Stock Trades/Day", value=int(config.get("MAX_STOCK_TRADES", 5)), step=1)
                new_nifty_limit = st.number_input("Max Nifty Trades/Day", value=int(config.get("MAX_NIFTY_TRADES", 2)), step=1)
                new_banknifty_limit = st.number_input("Max BankNifty Trades/Day", value=int(config.get("MAX_BANKNIFTY_TRADES", 2)), step=1)
                
            if st.button("💾 Save Risk Configuration"):
                new_config = config.copy()
                new_config["CAPITAL"] = new_capital
                new_config["ALLOCATION_PER_TRADE"] = new_allocation
                new_config["RISK_PER_TRADE_PERCENT"] = new_risk_pct
                new_config["MAX_DAILY_LOSS_PERCENT"] = new_max_loss
                new_config["MAX_STOCK_TRADES"] = new_stock_limit
                new_config["MAX_NIFTY_TRADES"] = new_nifty_limit
                new_config["MAX_BANKNIFTY_TRADES"] = new_banknifty_limit
                new_config["MAX_DAILY_TRADES"] = new_stock_limit + new_nifty_limit + new_banknifty_limit
                
                save_config(new_config)
                st.success("Configuration Saved! Bot will pick up changes on next trade check.")

        # --- Error Monitor ---
        if os.path.exists("bot.log"):
            with open("bot.log", "r") as f:
                logs = f.readlines()
                # Check last 10 lines for ERROR
                recent_logs = logs[-10:]
                error_found = False
                latest_error = ""
                for line in recent_logs:
                    if "ERROR" in line:
                        error_found = True
                        latest_error = line.strip()
                
                if error_found:
                    st.error(f"⚠️ Bot Reported an Error Recently:\n{latest_error}")

        st.subheader("📝 Live Logs (bot.log)")
        if os.path.exists("bot.log"):
            with open("bot.log", "r") as f:
                logs = f.readlines()
                # Show last 20 lines
                st.code("".join(logs[-20:]))
        else:
            st.info("No logs found yet.")
            
        st.subheader("📊 Trade Log")
        if os.path.exists("trade_log.csv"):
            trades_df = pd.read_csv("trade_log.csv")
            st.dataframe(trades_df)
        else:
            st.info("No trades executed yet.")
            
    # --- PAPER TRADING P&L ---
    elif page == "📜 Paper Trading P&L":
        st.header("📜 Paper Trading P&L Statement")
        st.markdown("*Real-time view of trades executed by the Daily Bot.*")
        st.markdown("---")
        
        trade_log_path = "trade_log.csv"
        
        # --- Auto Refresh for Live PnL ---
        if 'auto_refresh' not in st.session_state: st.session_state.auto_refresh = False
        
        col_header, col_refresh = st.columns([3,1])
        with col_header:
            st.header("📜 Paper Trading P&L")
        with col_refresh:
            st.session_state.auto_refresh = st.checkbox("🔴 Live Updates (10s)", value=st.session_state.auto_refresh)
            
        if st.session_state.auto_refresh:
            pass # Moved to bottom


        
        if os.path.exists(trade_log_path):
            try:
                df = pd.read_csv(trade_log_path)
                if not df.empty:
                    # Sort by newest first
                    if 'Time' in df.columns:
                        df['Time'] = pd.to_datetime(df['Time'])
                        # Filter for TODAY only
                        today_date = datetime.now().date()
                        df = df[df['Time'].dt.date == today_date]
                        
                        df.sort_values(by='Time', ascending=False, inplace=True)
                    
                    if df.empty:
                        st.info("No trades executed today yet.")
                        total_trades = 0
                        realized_pnl = 0.0
                    else:
                        # Metrics
                        total_trades = len(df)
                        realized_pnl = df[df['Status'] == 'CLOSED']['PnL'].sum() if 'PnL' in df.columns else 0.0
                    
                    # --- DYNAMIC PNL CALCULATION ---
                    # Identify Open Trades
                    open_trades_idx = df[df['Status'] == 'OPEN'].index
                    unrealized_pnl = 0.0
                    
                    if not open_trades_idx.empty:
                        open_tickers = df.loc[open_trades_idx, 'Ticker'].unique().tolist()

                        current_prices = {}
                        used_source = "Yahoo (Delayed)"
                        
                        # 1. Try Fyers (Real-Time)
                        if 'fyers_token' in st.session_state and st.session_state['fyers_token']:
                            try:
                                # Init Fyers
                                fyers = FyersApp(
                                    client_id=st.session_state.get('fyers_client_id', os.getenv('FYERS_CLIENT_ID')), 
                                    secret_key="secret", 
                                    access_token=st.session_state['fyers_token']
                                )
                                
                                # Convert Tickers to Fyers Format (NSE:TICKER-EQ)
                                fyers_symbols = []
                                fyers_map = {} # Map 'NSE:TICKER-EQ' back to 'TICKER.NS' or 'TICKER'
                                
                                for t in open_tickers:
                                    # Heuristic for symbol mapping
                                    clean_t = t.replace('.NS', '')
                                    if clean_t.startswith('^'): # Indices
                                        if clean_t == '^NSEI': fy = "NSE:NIFTY50-INDEX"
                                        elif clean_t == '^NSEBANK': fy = "NSE:NIFTYBANK-INDEX"
                                        else: fy = f"NSE:{clean_t}-INDEX"
                                    else:
                                        fy = f"NSE:{clean_t}-EQ"
                                    
                                    fyers_symbols.append(fy)
                                    fyers_map[fy] = t
                                
                                quotes = fyers.get_quotes(fyers_symbols)
                                
                                if quotes:
                                    for q in quotes:
                                        sym = q.get('n')
                                        lp = q.get('v', {}).get('lp') # LTP
                                        if sym in fyers_map and lp:
                                            current_prices[fyers_map[sym]] = float(lp)
                                    used_source = "Fyers (Live ⚡)"
                            except Exception as e:
                                st.error(f"Fyers Real-Time Error: {str(e)}")
                                st.warning("Falling back to Yahoo Finance...")

                        # 2. Fallback to Yahoo if needed
                        remaining_tickers = [t for t in open_tickers if t not in current_prices]
                        if remaining_tickers:
                            try:
                                tickers_str = " ".join(remaining_tickers)
                                live_data = yf.download(tickers_str, period="1d", interval="1m", progress=False)
                                
                                # Extract prices logic (Same as before)
                                if len(remaining_tickers) == 1:
                                    if not live_data.empty:
                                        price = live_data['Close'].iloc[-1]
                                        if isinstance(price, pd.Series): price = price.iloc[0]
                                        current_prices[remaining_tickers[0]] = float(price)
                                else:
                                    if not live_data.empty:
                                        last_row = live_data['Close'].iloc[-1]
                                        for t in remaining_tickers:
                                            if t in last_row:
                                                current_prices[t] = float(last_row[t])
                            except Exception as ex:
                                st.warning(f"Yahoo Fetch Failed: {ex}")

                        st.caption(f"🔄 Fetching prices via: **{used_source}**")

                        try:
                            # Update Dataframe
                            for idx in open_trades_idx:
                                ticker = df.at[idx, 'Ticker']

                                entry = float(df.at[idx, 'Entry_Price'])
                                qty = int(df.at[idx, 'Qty'])
                                signal = df.at[idx, 'Signal']
                                
                                if ticker in current_prices:
                                    cmp = current_prices[ticker]
                                    df.at[idx, 'Exit_Price'] = cmp # Show CMP in Exit Price for Open trades? Or separate? 
                                    # Let's use a new visual column 'CMP' and leave Exit Price as None (persisted)
                                    df.at[idx, 'CMP'] = cmp
                                    
                                    # Calc PnL
                                    pnl = 0.0
                                    if "LONG" in signal:
                                        pnl = (cmp - entry) * qty
                                    elif "SHORT" in signal:
                                        pnl = (entry - cmp) * qty
                                        
                                    df.at[idx, 'PnL'] = pnl
                                    unrealized_pnl += pnl
                                    
                                    # Mark as Unrealized in Reason for clarity?
                                    df.at[idx, 'Exit_Reason'] = "OPEN (Live)"
                                    
                        except Exception as e:
                            st.warning(f"Could not fetch live prices: {e}")

                    total_pnl = realized_pnl + unrealized_pnl

                    # Count wins/loss if PnL available
                    wins = len(df[df['PnL'] > 0]) if 'PnL' in df.columns else 0
                    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Trades", total_trades)
                    
                    # Dynamic Label with Live Indicator
                    pnl_label = "Today's Net P&L (Live)"
                    col2.metric(pnl_label, f"₹{total_pnl:,.2f}", delta=f"{total_pnl:.2f}")
                    
                    col3.metric("Realized", f"₹{realized_pnl:,.2f}")
                    col4.metric("Unrealized", f"₹{unrealized_pnl:,.2f}", delta_color="off")
                    
                    st.subheader("📝 Trade Log Details")
                    
                    # Ensure columns order if available
                    # Add CMP if we created it
                    display_cols = ['Time', 'Ticker', 'Signal', 'Entry_Price', 'CMP', 'Qty', 'PnL', 'Exit_Reason', 'Status']
                    
                    # Fill CMP for closed trades with Exit Price if missing
                    if 'CMP' not in df.columns: df['CMP'] = df['Exit_Price']
                    else: df['CMP'].fillna(df['Exit_Price'], inplace=True)

                    # Filter existing columns
                    final_cols = [c for c in display_cols if c in df.columns]
                    
                    # Fallback to all if schema mismatch
                    if not final_cols: final_cols = df.columns
                    
                    # Style PnL
                    st.dataframe(df[final_cols].style.format({"Entry_Price": "{:.2f}", "CMP": "{:.2f}", "PnL": "{:.2f}"}), use_container_width=True)
                else:
                    st.info("Trade Log exists but is empty. Waiting for trades...")
            except Exception as e:
                st.error(f"Error reading trade logs: {e}")
        else:
            st.info("No trade log found.")

        # --- Auto Refresh Execution ---
        if st.session_state.auto_refresh:
            time.sleep(10)
            st.rerun()

    elif page == "📡 Intraday Scanner":
        st.header("📡 Live Intraday Scanner")
        st.markdown("*Scans Nifty 50 for CPR Trends and Intraday Setups.*")
        
        if st.button("🔎 Run Live Scan", type="primary"):
            with st.spinner("Scanning Nifty 50..."):
                # Run scanner script and capture output? Or import logic?
                # Using subprocess to run independent script ensures clean state
                # But we can also replicate logic.
                # Let's run the script and read the CSV it produces.
                subprocess.run(["python3", "market_scanner.py"])
                
            st.success("Scan Completed!")
            
        if os.path.exists("daily_scan_results.csv"):
            st.subheader("Stocks in Play")
            df = pd.read_csv("daily_scan_results.csv")
            
            # Format
            def highlight_status(val):
                color = 'green' if 'Long' in val else 'red' if 'Short' in val else 'black'
                return f'color: {color}'
            
            st.dataframe(df.style.applymap(highlight_status, subset=['Status']), use_container_width=True)
        else:
            st.info("Run the scan to see results.")

    elif page == "⚡ Backtest Day Strategy":
        st.header("⚡ Backtest Day Trading Strategy")
        
        # Risk Settings
        with st.expander("⚙️ Capital & Risk Settings", expanded=True):
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                initial_cap = st.number_input("Initial Capital (₹)", value=100000, step=10000)
            with col_r2: 
                risk_pct = st.number_input("Risk Per Trade (%)", value=1.0, step=0.1, format="%.1f") / 100
            with col_r3:
                max_loss_pct = st.number_input("Max Daily Loss (%)", value=2.0, step=0.1, format="%.1f") / 100

        ticker = st.selectbox("Select Stock", ['^NSEI', '^NSEBANK'] + NIFTY_50)
        
        if st.button("Run Simulation"):
            with st.spinner(f"Backtesting {ticker} with {risk_pct*100}% Risk..."):
                results = backtest_day_strategy(
                    ticker, 
                    initial_capital=initial_cap,
                    risk_per_trade=risk_pct,
                    max_daily_loss=max_loss_pct
                )
                
            if results:
                st.success("Backtest Complete")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Win Rate", f"{results['win_rate']:.2f}%")
                col2.metric("Final Capital", f"₹{results['final_capital']:.2f}")
                col3.metric("Total Trades", len(results['trades']))
                
                if results['trades']:
                    df_trades = pd.DataFrame(results['trades'])
                    
                    # Weekday Analysis
                    if results.get('weekday_stats') is not None:
                        st.subheader("📅 Performance by Weekday")
                        st.dataframe(results['weekday_stats'].style.format({'PnL': '₹{:.2f}', 'Win Rate %': '{:.2f}%'}))

                    st.dataframe(df_trades)
                    
                    # Plot Chart
                    st.subheader("📉 Trade Visualization")
                    if 'data' in results:
                        chart = plot_day_trading_chart(results['data'], results['trades'], ticker)
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.warning("Chart data not available.")
    
    # --- SWING TRADING PAGES (Original Logic) ---




    elif page == "⚡ Backtest Indices (Options)":
        st.header("⚡ Backtest Index Options (Simulated)")
        st.markdown("*Simulates ATM Option buying (approx Delta 0.5) on Breakouts.*")
        
        # Risk Settings
        with st.expander("⚙️ Capital & Risk Settings", expanded=True):
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                initial_cap = st.number_input("Initial Capital (₹)", value=100000, step=10000, key="opt_cap")
            with col_r2: 
                risk_pct = st.number_input("Risk Per Trade (%)", value=1.0, step=0.1, format="%.1f", key="opt_risk") / 100
            with col_r3:
                max_loss_pct = st.number_input("Max Daily Loss (%)", value=2.0, step=0.1, format="%.1f", key="opt_max") / 100
            
        col_ex1, col_ex2 = st.columns([1, 1])
        with col_ex1:
            strategy_mode = st.radio("Strategy Mode", ['SNIPER', 'SURFER'])
        with col_ex2:
            exclude_days = st.multiselect("Exclude Trading Days", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], default=[])

        st.divider()

        ticker = st.selectbox("Select Index", ['^NSEI', '^NSEBANK'])
        
        if st.button("Run Option Simulation"):
            with st.spinner(f"Backtesting Options on {ticker} in {strategy_mode} Mode..."):
                results = backtest_day_strategy(
                    ticker, 
                    initial_capital=initial_cap,
                    risk_per_trade=risk_pct,
                    max_daily_loss=max_loss_pct,
                    strategy_mode=strategy_mode,
                    exclude_days=exclude_days
                )
                
            if results:
                st.success("Option Backtest Complete")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Win Rate", f"{results['win_rate']:.2f}%")
                col2.metric("Final Capital", f"₹{results['final_capital']:.2f}")
                col3.metric("Total Trades", len(results['trades']))
                
                if results['trades']:
                    df_trades = pd.DataFrame(results['trades'])
                    
                    # Weekday Analysis
                    if results.get('weekday_stats') is not None:
                        st.subheader("📅 Performance by Weekday")
                        st.dataframe(results['weekday_stats'].style.format({'PnL': '₹{:.2f}', 'Win Rate %': '{:.2f}%'}))

                    st.dataframe(df_trades)
                    
                    # Plot Chart
                    st.subheader("📉 Trade Visualization (Index Spot)")
                    if 'data' in results:
                        chart = plot_day_trading_chart(results['data'], results['trades'], ticker)
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.warning("Chart data not available.")

    elif page == "🏭 Sector Analysis":
        st.header("Sector Analysis Dashboard")
        st.markdown("""
        *Analyze sector-wise performance and identify which sectors to focus on for swing trading*

        **What this shows:**
        - Quarterly (3-month) returns by sector
        - Weekly and monthly momentum
        - Top performing stocks in each sector
        - Recommended sectors for swing trades
        """)

        if st.button("📊 Run Sector Analysis", type="primary"):
            if not nifty200:
                st.error("No stocks found in nifty200.csv")
            else:
                with st.spinner(f"Analyzing {len(nifty200)} stocks across sectors... This may take a minute."):
                    progress_bar = st.progress(0)

                    # Calculate sector performance
                    result = calculate_sector_performance(nifty200, period='3mo')
                    progress_bar.progress(100)
                    progress_bar.empty()

                    sector_perf = result['sector_performance']
                    stock_returns = result['stock_returns']

                if not sector_perf:
                    st.warning("Could not calculate sector performance. Please try again.")
                else:
                    # Store in session state for later use
                    st.session_state['sector_data'] = result

                    st.success(f"Analysis complete! Found {len(sector_perf)} sectors with {len(stock_returns)} stocks.")

                    # =====================================
                    # FOCUS SECTORS FOR SWING TRADING
                    # =====================================
                    st.markdown("---")
                    st.subheader("🎯 Recommended Sectors for Swing Trading")

                    focus_sectors = get_swing_focus_sectors(sector_perf, top_n=5)

                    if focus_sectors:
                        focus_cols = st.columns(len(focus_sectors))
                        for i, sector in enumerate(focus_sectors):
                            with focus_cols[i]:
                                momentum = sector['momentum']
                                rec = sector['recommendation']
                                color = "🟢" if rec == 'Focus' else "🟡"

                                st.markdown(f"### {color} {sector['sector']}")
                                st.metric("3M Return", f"{sector['avg_3m_return']:.1f}%")
                                st.metric("1W Return", f"{sector['avg_1w_return']:+.1f}%")
                                st.caption(f"{momentum}")
                                st.caption(f"Stocks: {sector['stock_count']}")

                        st.markdown("---")
                        st.info("**Trading Tip:** Focus on sectors with 'Bullish' momentum. Look for stocks within these sectors that show pullbacks to SMA21 for entry.")
                    else:
                        st.warning("No strong sectors identified currently. Market may be in consolidation.")

                    # =====================================
                    # SECTOR PERFORMANCE TABLE
                    # =====================================
                    st.markdown("---")
                    st.subheader("📊 Sector Performance Rankings")

                    # Create sector table
                    sector_df = pd.DataFrame([{
                        'Sector': s['sector'],
                        'Stocks': s['stock_count'],
                        '1 Week': f"{s['avg_1w_return']:+.1f}%",
                        '1 Month': f"{s['avg_1m_return']:+.1f}%",
                        '3 Months': f"{s['avg_3m_return']:+.1f}%",
                        'Best Stock (3M)': f"{s['best_3m']:+.1f}%",
                        'Momentum': get_sector_momentum_score(s)
                    } for s in sector_perf])

                    st.dataframe(sector_df, use_container_width=True, hide_index=True)

                    # =====================================
                    # SECTOR HEATMAP
                    # =====================================
                    st.markdown("---")
                    st.subheader("🗺️ Sector Performance Heatmap")

                    # Create heatmap data
                    heatmap_data = []
                    for s in sector_perf:
                        heatmap_data.append({
                            'Sector': s['sector'],
                            '1W': s['avg_1w_return'],
                            '1M': s['avg_1m_return'],
                            '3M': s['avg_3m_return']
                        })

                    heatmap_df = pd.DataFrame(heatmap_data)
                    heatmap_df = heatmap_df.set_index('Sector')

                    # Create heatmap using plotly
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_df.values,
                        x=['1 Week', '1 Month', '3 Months'],
                        y=heatmap_df.index,
                        colorscale='RdYlGn',
                        zmid=0,
                        text=[[f"{val:.1f}%" for val in row] for row in heatmap_df.values],
                        texttemplate="%{text}",
                        textfont={"size": 12},
                        hoverongaps=False
                    ))

                    fig.update_layout(
                        title='Sector Returns Heatmap',
                        template='plotly_dark',
                        height=max(400, len(sector_perf) * 25),
                        xaxis_title='Period',
                        yaxis_title='Sector'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # =====================================
                    # TOP STOCKS BY SECTOR
                    # =====================================
                    st.markdown("---")
                    st.subheader("🏆 Top Performing Stocks by Sector")

                    # Let user select a sector
                    selected_sector = st.selectbox(
                        "Select Sector to View Top Stocks",
                        [s['sector'] for s in sector_perf]
                    )

                    if selected_sector:
                        top_stocks = get_top_stocks_by_sector(stock_returns, selected_sector, top_n=10, period='3m')

                        if top_stocks:
                            st.markdown(f"**Top 10 stocks in {selected_sector} (by 3M return):**")

                            top_df = pd.DataFrame([{
                                'Stock': s['ticker'],
                                'Price': f"₹{s['current_price']:.2f}",
                                '1W Return': f"{s['1w_return']:+.1f}%",
                                '1M Return': f"{s['1m_return']:+.1f}%",
                                '3M Return': f"{s['3m_return']:+.1f}%"
                            } for s in top_stocks])

                            st.dataframe(top_df, use_container_width=True, hide_index=True)

                            # Quick add to watchlist
                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if st.button(f"➕ Add Top 5 to Watchlist"):
                                    for s in top_stocks[:5]:
                                        if s['ticker'] not in st.session_state.watchlist:
                                            st.session_state.watchlist.append(s['ticker'])
                                    st.success(f"Added top 5 {selected_sector} stocks to Watchlist!")
                        else:
                            st.warning(f"No stocks found for {selected_sector}")

                    # =====================================
                    # ALL STOCKS PERFORMANCE
                    # =====================================
                    st.markdown("---")
                    st.subheader("📋 All Stock Returns (Sorted by 3M)")

                    # Sort all stocks by 3M return
                    all_stocks_sorted = sorted(stock_returns, key=lambda x: x.get('3m_return', 0), reverse=True)

                    all_stocks_df = pd.DataFrame([{
                        'Stock': s['ticker'],
                        'Sector': s['sector'],
                        'Price': f"₹{s['current_price']:.2f}",
                        '1W': f"{s['1w_return']:+.1f}%",
                        '1M': f"{s['1m_return']:+.1f}%",
                        '3M': f"{s['3m_return']:+.1f}%"
                    } for s in all_stocks_sorted[:50]])  # Top 50

                    st.dataframe(all_stocks_df, use_container_width=True, hide_index=True)

                    # Download option
                    full_df = pd.DataFrame([{
                        'Stock': s['ticker'],
                        'Sector': s['sector'],
                        'Price': s['current_price'],
                        '1W_Return': s['1w_return'],
                        '1M_Return': s['1m_return'],
                        '3M_Return': s['3m_return']
                    } for s in all_stocks_sorted])

                    csv = full_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Sector Analysis CSV",
                        csv,
                        "sector_analysis.csv",
                        "text/csv"
                    )

                    # =====================================
                    # TRADING RECOMMENDATIONS
                    # =====================================
                    st.markdown("---")
                    st.subheader("💡 Swing Trading Recommendations")

                    if focus_sectors:
                        st.markdown("Based on sector analysis, here are the recommendations:")

                        for i, sector in enumerate(focus_sectors[:3], 1):
                            with st.expander(f"#{i} {sector['sector']} - {sector['momentum']}", expanded=(i==1)):
                                st.markdown(f"""
                                **Sector Stats:**
                                - 3-Month Return: **{sector['avg_3m_return']:.1f}%**
                                - 1-Month Return: {sector['avg_1m_return']:+.1f}%
                                - 1-Week Return: {sector['avg_1w_return']:+.1f}%
                                - Number of Stocks: {sector['stock_count']}

                                **Trading Strategy:**
                                """)

                                if 'Bullish' in sector['momentum']:
                                    st.success("""
                                    ✅ **FOCUS SECTOR** - Look for pullback entries
                                    - Wait for stocks to pull back to SMA21
                                    - Enter when RSI is between 40-55
                                    - Use the Stock Screener to find specific signals
                                    """)
                                else:
                                    st.warning("""
                                    👁️ **WATCH SECTOR** - Wait for better entry
                                    - Momentum is weakening
                                    - Consider waiting for RSI oversold bounce
                                    - Monitor for trend reversal
                                    """)

                                # Show top 3 stocks from this sector
                                sector_top = get_top_stocks_by_sector(stock_returns, sector['sector'], top_n=3, period='3m')
                                if sector_top:
                                    st.markdown("**Top Stocks to Watch:**")
                                    for s in sector_top:
                                        st.markdown(f"- **{s['ticker']}**: ₹{s['current_price']:.2f} | 3M: {s['3m_return']:+.1f}%")

                    else:
                        st.warning("""
                        **No strong sectors currently.**

                        Market appears to be in consolidation. Consider:
                        - Reducing position sizes
                        - Focusing on individual stock setups rather than sector plays
                        - Waiting for clearer sector trends to emerge
                        """)

        # Show cached data if available
        elif 'sector_data' in st.session_state:
            st.info("💾 Showing cached sector data. Click 'Run Sector Analysis' to refresh.")

            result = st.session_state['sector_data']
            sector_perf = result['sector_performance']
            stock_returns = result['stock_returns']

            # Display cached sector table
            st.subheader("📊 Cached Sector Performance")
            sector_df = pd.DataFrame([{
                'Sector': s['sector'],
                'Stocks': s['stock_count'],
                '1 Week': f"{s['avg_1w_return']:+.1f}%",
                '1 Month': f"{s['avg_1m_return']:+.1f}%",
                '3 Months': f"{s['avg_3m_return']:+.1f}%",
                'Momentum': get_sector_momentum_score(s)
            } for s in sector_perf])

            st.dataframe(sector_df, use_container_width=True, hide_index=True)

if __name__ == '__main__':
    main()
