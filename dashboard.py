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
from datetime import datetime, timedelta
import os

from swing_strategy import (
    fetch_ohlc_extended, add_all_indicators, detect_swing_signal,
    screen_swing_signals, backtest_strategy, calculate_risk_metrics,
    calculate_signal_strength, detect_weekly_breakout_pullback,
    screen_weekly_breakout, screen_52week_high, screen_high_volume_trend, calculate_sector_performance, get_sector,
    get_sector_momentum_score, get_swing_focus_sectors, get_top_stocks_by_sector,
    get_all_sectors
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
    default_app_id = st.secrets.get("FYERS_APP_ID", "")
    default_secret = st.secrets.get("FYERS_SECRET_ID", "")
    
    fyers_app_id = st.text_input("Client ID (App ID)", value=default_app_id, help="e.g., XCXXXXX-100")
    fyers_secret = st.text_input("Secret Key", value=default_secret, type="password")
    fyers_redirect = st.text_input("Redirect URI", value="https://trade.fyers.in/api-login/redirect-uri/index.html")

    if st.button("Generate Login Link"):
        fyers_app_id = fyers_app_id.strip()
        fyers_secret = fyers_secret.strip()
        fyers_redirect = fyers_redirect.strip()
        
        if fyers_app_id and fyers_secret:
            app = FyersApp(fyers_app_id, fyers_secret, redirect_uri=fyers_redirect)
            auth_url = app.get_login_url()
            st.markdown(f"[**👉 Click to Login**]({auth_url})", unsafe_allow_html=True)
            st.info("1. Click Link & Login\n2. Copy 'auth_code' from URL\n3. Paste below")
        else:
            st.error("Enter App ID & Secret")

    auth_code_input = st.text_input("Paste Auth Code Here (or full 404 URL)")
    
    if st.button("Authenticate & Connect"):
        fyers_app_id = fyers_app_id.strip()
        fyers_secret = fyers_secret.strip()
        
        # Smart Extraction: Handle full URL paste
        if "auth_code=" in auth_code_input:
            try:
                import urllib.parse
                parsed = urllib.parse.urlparse(auth_code_input)
                params = urllib.parse.parse_qs(parsed.query)
                if 'auth_code' in params:
                    auth_code_input = params['auth_code'][0]
                    st.success(f"✅ Extracted Code: {auth_code_input[:10]}...{auth_code_input[-5:]}")
            except:
                pass 
        
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
                st.error(f"Login Failed: {e}")
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
    st.title("📈 Nifty 200 Swing Trading App")
    st.markdown("*Professional Multi-Indicator Strategy with Backtesting*")
    st.markdown("---")
    # Sidebar
    st.sidebar.title("Navigation")
    
    st.sidebar.markdown("### Swing Trading")
    swing_page = st.sidebar.radio(
        "Swing Modules",
        ["🔍 Stock Screener", "📅 Weekly Breakout", "🚀 52-Week High", "📊 High Volume", "📈 Stock Analysis", "👁️ Watchlist", "🏭 Sector Analysis", "🧪 Backtest Swing", "💰 Risk Calculator"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Day Trading Robot")
    dt_page = st.sidebar.radio(
        "Day Trading Modules",
        ["🤖 Live Bot Manager", "📡 Intraday Scanner", "⚡ Backtest Day Strategy", "⚡ Backtest Indices (Options)"]
    )
    
    # Logic to determine active page
    # If sidebar radio key is needed, we manage state. Simpler to just check which radio was last clicked or default.
    # Streamlit re-runs, so we need to know which section was interacted with.
    # Hack: We act on the one that is NOT default if possible, or use Session State to toggle mode.
    # Assuming user clicks one.
    
    # We combine them into a single 'page' variable for the main logic
    # But Streamlit radios are independent.
    # Visual separation is good. Let's use a SelectBox for Main Mode?
    
    mode = st.sidebar.selectbox("Select Mode", ["Swing Trading", "Day Trading Robot"])
    
    page = ""
    if mode == "Swing Trading":
        page = swing_page
    else:
        page = dt_page

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
            cmd = "ps -ef | grep 'live_bot.py' | grep -v grep"
            output = subprocess.check_output(cmd, shell=True).decode()
            if 'live_bot.py' in output:
                bot_running = True
        except:
            bot_running = False
            
        # AI Configuration
        with st.expander("🤖 AI Configuration (Gemini)", expanded=False):
            default_gemini = st.secrets.get("GEMINI_API_KEY", "")
            api_key = st.text_input("Gemini API Key", value=default_gemini, type="password", key="gemini_key")
            if api_key:
                st.session_state['gemini_api_key'] = api_key
                st.caption("✅ Key saved for this session.")
            else:
                st.caption("Enter key to enable AI Validation.")

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

    if page == "🔍 Stock Screener":
        st.header("Stock Screener - Multi-Indicator Analysis")

        # Show confirmation message if watchlist was just updated (at the TOP)
        if st.session_state.get('watchlist_updated'):
            st.success(f"✅ **Watchlist updated! Now has {len(st.session_state.watchlist)} stocks.** Go to 👁️ Watchlist tab to analyze.")
            st.session_state['watchlist_updated'] = False

        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("Minimum Signal Strength", 30, 80, 50)
        with col2:
            max_results = st.slider("Max Results", 5, 50, 20)
        with col3:
            period = st.selectbox("Analysis Period", ['6mo', '1y', '2y'], index=1)

        if st.button("🔍 Run Screener", type="primary"):
            if not nifty200:
                st.error("No stocks found in nifty200.csv")
            else:
                with st.spinner(f"Scanning {len(nifty200)} Nifty 200 stocks..."):
                    progress_bar = st.progress(0)
                    results = []

                    for i, ticker in enumerate(nifty200):
                        try:
                            df = fetch_ohlc_extended(ticker, period=period)
                            if df.empty or len(df) < 50:
                                continue

                            df = add_all_indicators(df)
                            signal = detect_swing_signal(df, min_score=min_score)

                            if signal.get('signal'):
                                results.append({
                                    'Ticker': ticker.replace('.NS', ''),
                                    'Close': signal['close'],
                                    'RSI': signal['rsi'],
                                    'ADX': signal['adx'],
                                    'Grade': signal['signal_grade'],
                                    'Score': signal['signal_strength'],
                                    'Stop Loss': signal['stop_loss'],
                                    'Target 1': signal['target1'],
                                    'Target 2': signal['target2'],
                                    'Risk %': signal['risk_percent']
                                })
                        except Exception:
                            continue

                        progress_bar.progress((i + 1) / len(nifty200))

                progress_bar.empty()

                if results:
                    results_df = pd.DataFrame(results)
                    results_df = results_df.sort_values('Score', ascending=False).head(max_results)
                    # Store results in session state so they persist
                    st.session_state.screener_results = results_df['Ticker'].tolist()
                    st.session_state['screener_results_df'] = results_df.to_dict('records')
                else:
                    st.warning("No stocks found matching the criteria.")
                    st.session_state['screener_results_df'] = []

        # Display results from session state (persists after rerun)
        if st.session_state.get('screener_results_df'):
            results_df = pd.DataFrame(st.session_state['screener_results_df'])

            st.success(f"Found {len(results_df)} stocks with signals!")

            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("📋 Add All to Watchlist", type="primary"):
                    added_count = 0
                    for ticker in st.session_state.screener_results:
                        if ticker not in st.session_state.watchlist:
                            st.session_state.watchlist.append(ticker)
                            added_count += 1
                    st.session_state['watchlist_updated'] = True
                    st.rerun()

            with col2:
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    "screener_results.csv",
                    "text/csv"
                )

            with col3:
                if st.session_state.watchlist:
                    if st.button("🗑️ Clear Watchlist"):
                        st.session_state.watchlist = []
                        st.session_state.trade_notes = {}
                        st.rerun()

            with col4:
                if st.session_state.watchlist:
                    st.success(f"✓ Watchlist: {len(st.session_state.watchlist)} stocks")
                else:
                    st.info("Watchlist: Empty")

            st.markdown("---")

            # Display results with clickable analysis
            st.subheader("📊 Screener Results - Click to Analyze")

            # Display as cards with buttons
            for idx, row in results_df.iterrows():
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1, 1, 1.5, 1.5])

                    with col1:
                        st.markdown(f"**{row['Ticker']}**")
                        st.caption(f"Grade: {row['Grade']} | Score: {row['Score']:.0f}")

                    with col2:
                        st.metric("Price", f"₹{row['Close']:.2f}")

                    with col3:
                        st.metric("RSI", f"{row['RSI']:.1f}")

                    with col4:
                        st.metric("Risk", f"{row['Risk %']:.1f}%")

                    with col5:
                        if st.button(f"📊 Analyze", key=f"analyze_{row['Ticker']}"):
                            st.session_state.selected_stock_for_analysis = row['Ticker']
                            if row['Ticker'] not in st.session_state.watchlist:
                                st.session_state.watchlist.append(row['Ticker'])
                            st.info(f"✅ {row['Ticker']} added to watchlist. Go to 📊 Stock Analysis or 👁️ Watchlist tab")

                    with col6:
                        if row['Ticker'] in st.session_state.watchlist:
                            st.success("✓ In Watchlist")
                        else:
                            if st.button(f"➕ Add", key=f"add_{row['Ticker']}"):
                                st.session_state.watchlist.append(row['Ticker'])
                                st.rerun()

                    st.markdown("---")

            # Summary table
            st.subheader("📋 Full Results Table")
            st.dataframe(
                results_df.style.format({
                    'Close': '₹{:.2f}',
                    'RSI': '{:.1f}',
                    'ADX': '{:.1f}',
                    'Score': '{:.0f}',
                    'Stop Loss': '₹{:.2f}',
                    'Target 1': '₹{:.2f}',
                    'Target 2': '₹{:.2f}',
                    'Risk %': '{:.2f}%'
                }).background_gradient(subset=['Score'], cmap='RdYlGn'),
                use_container_width=True
            )

    elif page == "📅 Weekly Breakout":
        st.header("Weekly High/Low Breakout Screener")
        st.markdown("""
        *Finds stocks that crossed their weekly high but pulled back - potential breakout re-entry setups*

        **Strategy Logic:**
        - Identifies previous week's high and low levels
        - Screens for stocks that crossed above weekly high (showed strength)
        - But closed below weekly high (pullback = better entry opportunity)
        """)

        if st.button("🔍 Run Weekly Breakout Screener", type="primary"):
            if not nifty200:
                st.error("No stocks found in nifty200.csv")
            else:
                with st.spinner(f"Scanning {len(nifty200)} stocks for weekly breakout setups..."):
                    progress_bar = st.progress(0)
                    results = []

                    for i, ticker in enumerate(nifty200):
                        try:
                            signal = detect_weekly_breakout_pullback(ticker)
                            if signal.get('signal'):
                                results.append(signal)
                        except Exception:
                            continue
                        progress_bar.progress((i + 1) / len(nifty200))

                    progress_bar.empty()

                if results:
                    # Sort by distance from weekly high
                    results.sort(key=lambda x: x.get('distance_from_weekly_high', 999))

                    st.success(f"Found {len(results)} stocks with weekly breakout-pullback setups!")

                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button("📋 Add All to Watchlist", key="weekly_add_all"):
                            for r in results:
                                if r['ticker'] not in st.session_state.watchlist:
                                    st.session_state.watchlist.append(r['ticker'])
                            st.session_state['watchlist_updated'] = True
                            st.rerun()

                    with col2:
                        results_df = pd.DataFrame(results)
                        csv = results_df.to_csv(index=False)
                        st.download_button("📥 Download CSV", csv, "weekly_breakout.csv", "text/csv")

                    with col3:
                        if st.session_state.watchlist:
                            if st.button("🗑️ Clear Watchlist", key="weekly_clear"):
                                st.session_state.watchlist = []
                                st.session_state.trade_notes = {}
                                st.rerun()

                    with col4:
                        if st.session_state.watchlist:
                            st.success(f"✓ Watchlist: {len(st.session_state.watchlist)} stocks")
                        else:
                            st.info("Watchlist: Empty")

                    # Show confirmation message if watchlist was just updated
                    if st.session_state.get('watchlist_updated'):
                        st.success(f"✅ Watchlist updated! Now has {len(st.session_state.watchlist)} stocks. Go to 👁️ Watchlist tab to analyze.")
                        st.session_state['watchlist_updated'] = False

                    st.markdown("---")

                    # Display results
                    st.subheader("📊 Weekly Breakout Setups")

                    for r in results:
                        with st.container():
                            col1, col2, col3, col4, col5, col6 = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 1])

                            with col1:
                                st.markdown(f"**{r['ticker']}**")
                                quality_color = "🟢" if r['setup_quality'].startswith('A') else "🟡" if r['setup_quality'].startswith('B') else "🟠" if r['setup_quality'].startswith('C') else "🔴"
                                st.caption(f"{quality_color} {r['setup_quality']}")

                            with col2:
                                st.metric("Close", f"₹{r['current_close']:.2f}")

                            with col3:
                                st.metric("Weekly High", f"₹{r['weekly_high']:.2f}")

                            with col4:
                                st.metric("Distance", f"-{r['distance_from_weekly_high']:.1f}%")

                            with col5:
                                st.metric("Stop Loss", f"₹{r['stop_loss']:.2f}")

                            with col6:
                                if r['ticker'] in st.session_state.watchlist:
                                    st.success("✓ Added")
                                else:
                                    if st.button("➕", key=f"wadd_{r['ticker']}"):
                                        st.session_state.watchlist.append(r['ticker'])
                                        st.rerun()

                            # Expandable details
                            with st.expander(f"View Details - {r['ticker']}"):
                                det_col1, det_col2, det_col3 = st.columns(3)
                                with det_col1:
                                    st.markdown("**Weekly Levels:**")
                                    st.write(f"High: ₹{r['weekly_high']:.2f}")
                                    st.write(f"Low: ₹{r['weekly_low']:.2f}")
                                    st.write(f"Range: ₹{r['weekly_range']:.2f} ({r['weekly_range_pct']:.1f}%)")

                                with det_col2:
                                    st.markdown("**Current Position:**")
                                    st.write(f"Close: ₹{r['current_close']:.2f}")
                                    st.write(f"From High: -{r['distance_from_weekly_high']:.2f}%")
                                    st.write(f"From Low: +{r['distance_from_weekly_low']:.2f}%")

                                with det_col3:
                                    st.markdown("**Trade Setup:**")
                                    st.write(f"Entry Zone: {r['entry_zone']}")
                                    st.write(f"Stop Loss: ₹{r['stop_loss']:.2f}")
                                    st.write(f"Target 1: ₹{r['target1']:.2f}")
                                    st.write(f"Target 2: ₹{r['target2']:.2f}")

                            st.markdown("---")

                    # Summary table
                    st.subheader("📋 Full Results Table")
                    display_df = pd.DataFrame([{
                        'Stock': r['ticker'],
                        'Close': r['current_close'],
                        'Weekly High': r['weekly_high'],
                        'Weekly Low': r['weekly_low'],
                        'Dist from High': f"-{r['distance_from_weekly_high']:.1f}%",
                        'Setup': r['setup_quality'],
                        'Stop Loss': r['stop_loss'],
                        'Target 1': r['target1']
                    } for r in results])

                    st.dataframe(
                        display_df.style.format({
                            'Close': '₹{:.2f}',
                            'Weekly High': '₹{:.2f}',
                            'Weekly Low': '₹{:.2f}',
                            'Stop Loss': '₹{:.2f}',
                            'Target 1': '₹{:.2f}'
                        }),
                        use_container_width=True
                    )

                else:
                    st.warning("No weekly breakout-pullback setups found currently.")
                    st.info("This screener looks for stocks that crossed their weekly high but pulled back. Try again after market hours or next week.")

    elif page == "🚀 52-Week High":
        st.header("52-Week High Screener")
        st.markdown("*Find stocks trading near their 52-week highs - strong momentum signals*")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            tolerance = st.slider("Distance from 52-week high (%)", min_value=0.1, max_value=5.0, value=2.0, step=0.1)
            st.caption(f"Show stocks within {tolerance}% of their 52-week high")
        
        with col2:
            st.metric("Market Status", "Open" if datetime.now().weekday() < 5 and 9 <= datetime.now().hour < 16 else "Closed")

        if st.button("🚀 Scan for 52-Week Highs", type="primary"):
            try:
                with st.spinner(f"Scanning {len(nifty200)} stocks near 52-week highs..."):
                    # Convert tickers to yfinance format for fetching
                    yf_tickers = [t.replace('NSE:', '').replace('-EQ', '') + '.NS' for t in nifty200]
                    results = screen_52week_high(yf_tickers, tolerance_pct=tolerance)

                if results:
                    st.session_state.screener_results = results
                    st.success(f"✅ Found {len(results)} stocks trading near 52-week highs!")

                    # Display results
                    df_display = pd.DataFrame([
                        {
                            'Stock': r['ticker'],
                            'Current Price': f"₹{r['current_price']}",
                            '52W High': f"₹{r['52week_high']}",
                            'Distance (%)': f"{r['distance_from_high_pct']}%",
                            'Strength': r['strength']
                        }
                        for r in results
                    ])

                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        height=min(500, len(results) * 35 + 100)
                    )

                    # Export option
                    csv = df_display.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"52week_highs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                    # Add to watchlist option
                    st.subheader("Add to Watchlist")
                    selected_stocks = st.multiselect(
                        "Select stocks to add to watchlist:",
                        options=[r['ticker'] for r in results],
                        default=[]
                    )
                    if st.button("➕ Add Selected to Watchlist"):
                        st.session_state.watchlist.extend(selected_stocks)
                        st.session_state.watchlist = list(set(st.session_state.watchlist))  # Remove duplicates
                        st.success(f"✅ Added {len(selected_stocks)} stocks to watchlist!")

                else:
                    st.warning("No stocks found near 52-week highs in the current tolerance range.")
                    st.info("Try increasing the tolerance percentage or check after market hours.")

            except Exception as e:
                st.error(f"❌ Error scanning: {str(e)}")
                st.info("Please try again or adjust the parameters.")

    elif page == "📊 High Volume":
        st.header("High Volume Trader Screener")
        st.markdown("*Find stocks with continuously high trading volume - indicates strong institutional interest*")
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            lookback_days = st.slider("Number of consecutive days", min_value=2, max_value=10, value=3, step=1)
            st.caption(f"Analyze volume for last {lookback_days} days")
        
        with col2:
            st.metric("Market Status", "Open" if datetime.now().weekday() < 5 and 9 <= datetime.now().hour < 16 else "Closed")

        if st.button("📊 Scan for High Volume Traders", type="primary"):
            try:
                with st.spinner(f"Scanning {len(nifty200)} stocks for high volume trends..."):
                    # Convert tickers to yfinance format
                    yf_tickers = [t.replace('NSE:', '').replace('-EQ', '') + '.NS' for t in nifty200]
                    results = screen_high_volume_trend(yf_tickers, lookback_days=lookback_days)

                if results:
                    st.session_state.screener_results = results
                    st.success(f"✅ Found {len(results)} stocks with strong volume trends!")

                    # Display results
                    df_display = pd.DataFrame([
                        {
                            'Stock': r['ticker'],
                            'Current Price': f"₹{r['current_price']}",
                            '3D Avg Volume': r['avg_volume_3d'],
                            'Volume Momentum (%)': f"{r['volume_momentum_pct']}%",
                            'Price Change (%)': f"{r['price_change_pct']}%",
                            'Strength': r['strength'].replace('_', ' ')
                        }
                        for r in results
                    ])

                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        height=min(500, len(results) * 35 + 100)
                    )

                    # Export option
                    csv = df_display.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"high_volume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                    # Add to watchlist option
                    st.subheader("Add to Watchlist")
                    selected_stocks = st.multiselect(
                        "Select stocks to add to watchlist:",
                        options=[r['ticker'] for r in results],
                        default=[]
                    )
                    if st.button("➕ Add Selected to Watchlist", key="add_volume_watchlist"):
                        st.session_state.watchlist.extend(selected_stocks)
                        st.session_state.watchlist = list(set(st.session_state.watchlist))  # Remove duplicates
                        st.success(f"✅ Added {len(selected_stocks)} stocks to watchlist!")

                else:
                    st.warning("No stocks found with high volume trends.")
                    st.info("Try adjusting the lookback period or check after market hours.")

            except Exception as e:
                st.error(f"❌ Error scanning: {str(e)}")
                st.info("Please try again or adjust the parameters.")

    elif page == "📈 Stock Analysis":
        st.header("Individual Stock Analysis")

        # Show watchlist stocks if available
        if st.session_state.watchlist:
            st.info(f"📋 Watchlist has {len(st.session_state.watchlist)} stocks: {', '.join(st.session_state.watchlist[:5])}{'...' if len(st.session_state.watchlist) > 5 else ''}")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            # Stock selection - prioritize watchlist stocks
            if nifty200:
                stock_options = [t.replace('.NS', '') for t in nifty200]

                # Set default to selected stock from screener or first watchlist stock
                default_idx = 0
                if st.session_state.selected_stock_for_analysis:
                    if st.session_state.selected_stock_for_analysis in stock_options:
                        default_idx = stock_options.index(st.session_state.selected_stock_for_analysis)
                elif st.session_state.watchlist:
                    if st.session_state.watchlist[0] in stock_options:
                        default_idx = stock_options.index(st.session_state.watchlist[0])

                selected = st.selectbox("Select Stock", stock_options, index=default_idx)
                ticker = f"{selected}.NS"
            else:
                ticker = st.text_input("Enter Ticker (e.g., RELIANCE.NS)", "RELIANCE.NS")

        with col2:
            period = st.selectbox("Period", ['3mo', '6mo', '1y', '2y'], index=2, key='analysis_period')

        with col3:
            # Quick switch between watchlist stocks
            if st.session_state.watchlist:
                st.markdown("**Quick Switch:**")
                watchlist_select = st.selectbox(
                    "Watchlist",
                    ["--Select--"] + st.session_state.watchlist,
                    key='quick_watchlist'
                )
                if watchlist_select != "--Select--":
                    st.session_state.selected_stock_for_analysis = watchlist_select
                    st.rerun()

        # Action buttons row
        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
        with btn_col1:
            analyze_clicked = st.button("📊 Analyze Stock", type="primary")
        with btn_col2:
            current_stock = selected if nifty200 else ticker.replace('.NS', '')
            if current_stock in st.session_state.watchlist:
                st.success("✓ In Watchlist")
            else:
                if st.button("➕ Add to Watchlist"):
                    st.session_state.watchlist.append(current_stock)
                    st.success(f"Added {current_stock} to watchlist!")
                    st.rerun()
        with btn_col3:
            if st.session_state.watchlist:
                st.caption(f"Watchlist: {', '.join(st.session_state.watchlist[:5])}{'...' if len(st.session_state.watchlist) > 5 else ''}")

        if analyze_clicked:
            with st.spinner(f"Analyzing {ticker}..."):
                df = fetch_ohlc_extended(ticker, period=period)

                if df.empty:
                    st.error(f"Could not fetch data for {ticker}")
                    return

                df = add_all_indicators(df)
                signal = detect_swing_signal(df, min_score=30)

                # Current values
                last = df.iloc[-1]

                st.markdown("---")

                # Key Metrics
                st.subheader("📈 Current Indicators")
                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("Close", f"₹{last['Close']:.2f}")
                with col2:
                    rsi_color = "inverse" if last['RSI'] > 70 else "normal" if last['RSI'] < 30 else "off"
                    st.metric("RSI", f"{last['RSI']:.1f}", delta_color=rsi_color)
                with col3:
                    st.metric("ADX", f"{last['ADX']:.1f}")
                with col4:
                    macd_delta = last['MACD'] - last['MACD_Signal']
                    st.metric("MACD", f"{last['MACD']:.2f}", f"{macd_delta:.2f}")
                with col5:
                    st.metric("ATR", f"₹{last['ATR']:.2f}")

                # SMA Status
                st.subheader("📊 Moving Average Status")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    above_sma21 = "✅ Above" if last['Close'] > last['SMA21'] else "❌ Below"
                    st.metric("vs SMA21", above_sma21, f"₹{last['SMA21']:.2f}")
                with col2:
                    above_sma50 = "✅ Above" if last['Close'] > last['SMA50'] else "❌ Below"
                    st.metric("vs SMA50", above_sma50, f"₹{last['SMA50']:.2f}")
                with col3:
                    above_sma200 = "✅ Above" if last['Close'] > last['SMA200'] else "❌ Below"
                    st.metric("vs SMA200", above_sma200, f"₹{last['SMA200']:.2f}")
                with col4:
                    trend = "🟢 Bullish" if last['SMA50'] > last['SMA200'] else "🔴 Bearish"
                    st.metric("Trend", trend)

                # Signal Analysis
                st.markdown("---")
                if signal.get('signal'):
                    st.success("✅ SWING SIGNAL DETECTED!")
                    strength = calculate_signal_strength(df)
                    display_signal_strength(strength)

                    st.subheader("🎯 Trade Setup")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Entry", f"₹{signal['close']:.2f}")
                    with col2:
                        st.metric("Stop Loss", f"₹{signal['stop_loss']:.2f}",
                                  f"-{signal['risk_percent']:.1f}%")
                    with col3:
                        t1_return = (signal['target1'] / signal['close'] - 1) * 100
                        st.metric("Target 1", f"₹{signal['target1']:.2f}", f"+{t1_return:.1f}%")
                    with col4:
                        t2_return = (signal['target2'] / signal['close'] - 1) * 100
                        st.metric("Target 2", f"₹{signal['target2']:.2f}", f"+{t2_return:.1f}%")
                else:
                    st.warning("⚠️ No active signal. Stock may not meet all criteria.")
                    # Still show strength
                    strength = calculate_signal_strength(df)
                    display_signal_strength(strength)

                # Chart
                st.markdown("---")
                st.subheader("📉 Technical Chart")
                chart = create_candlestick_chart(df.tail(120), ticker)
                st.plotly_chart(chart, use_container_width=True)

    elif page == "👁️ Watchlist":
        st.header("Watchlist - Manual Trade Selection")
        st.markdown("*Select multiple stocks to analyze charts and decide trades manually*")

        # Show current watchlist count prominently
        if st.session_state.watchlist:
            st.success(f"✅ **Watchlist has {len(st.session_state.watchlist)} stocks:** {', '.join(st.session_state.watchlist[:10])}{'...' if len(st.session_state.watchlist) > 10 else ''}")
        else:
            # Show if stocks came from screener but not yet added
            if st.session_state.screener_results:
                st.warning(f"📊 Screener found {len(st.session_state.screener_results)} stocks. Go back to Screener tab and click 'Add All to Watchlist' to add them here.")
            else:
                st.info("👆 Add stocks using the dropdown below or run the Stock Screener first.")

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            if nifty200:
                stock_options = [t.replace('.NS', '') for t in nifty200]
                # Use a key to make the multiselect work properly with session state
                selected_stocks = st.multiselect(
                    "Add/Remove stocks from watchlist",
                    stock_options,
                    default=st.session_state.watchlist,
                    help="Select multiple stocks to compare charts",
                    key="watchlist_multiselect"
                )
                # Only update if user actually changed the selection
                if selected_stocks != st.session_state.watchlist:
                    st.session_state.watchlist = selected_stocks
                    st.rerun()

        with col2:
            period = st.selectbox("Chart Period", ['1mo', '3mo', '6mo', '1y'], index=2, key='watchlist_period')

        with col3:
            if st.session_state.watchlist:
                if st.button("🗑️ Clear All", type="secondary"):
                    st.session_state.watchlist = []
                    st.session_state.trade_notes = {}
                    st.rerun()

        # Quick add from popular stocks
        st.markdown("**Quick Add Popular Stocks:**")
        popular_cols = st.columns(8)
        popular_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'BHARTIARTL', 'ITC', 'SBIN']
        for i, stock in enumerate(popular_stocks):
            with popular_cols[i]:
                if st.button(stock, key=f'quick_{stock}'):
                    if stock not in st.session_state.watchlist:
                        st.session_state.watchlist.append(stock)
                        st.rerun()

        st.markdown("---")

        if st.session_state.watchlist:
            st.subheader(f"📊 Analyzing {len(st.session_state.watchlist)} Stocks")

            # Summary table
            summary_data = []
            for stock in st.session_state.watchlist:
                ticker = f"{stock}.NS"
                try:
                    df = fetch_ohlc_extended(ticker, period=period)
                    if df.empty or len(df) < 20:
                        continue
                    df = add_all_indicators(df)
                    last = df.iloc[-1]
                    prev = df.iloc[-2]

                    # Price change
                    price_change = (last['Close'] - prev['Close']) / prev['Close'] * 100

                    # Trend
                    trend = "🟢 Bullish" if last['Close'] > last['SMA50'] > last['SMA200'] else "🔴 Bearish" if last['Close'] < last['SMA50'] < last['SMA200'] else "🟡 Sideways"

                    # Signal check
                    signal = detect_swing_signal(df, min_score=40)
                    signal_status = f"✅ {signal.get('signal_grade', 'N/A')}" if signal.get('signal') else "❌ No Signal"

                    summary_data.append({
                        'Stock': stock,
                        'Price': f"₹{last['Close']:.2f}",
                        'Change': f"{price_change:+.2f}%",
                        'RSI': f"{last['RSI']:.1f}",
                        'Trend': trend,
                        'Signal': signal_status,
                        'vs SMA21': f"{((last['Close']-last['SMA21'])/last['SMA21']*100):+.1f}%"
                    })
                except:
                    continue

            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

            st.markdown("---")

            # Individual charts with notes
            for stock in st.session_state.watchlist:
                ticker = f"{stock}.NS"
                try:
                    df = fetch_ohlc_extended(ticker, period=period)
                    if df.empty or len(df) < 20:
                        continue
                    df = add_all_indicators(df)

                    with st.expander(f"📈 {stock} - Click to expand chart", expanded=False):
                        # Quick stats
                        last = df.iloc[-1]
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Close", f"₹{last['Close']:.2f}")
                        with col2:
                            st.metric("RSI", f"{last['RSI']:.1f}")
                        with col3:
                            st.metric("SMA21", f"₹{last['SMA21']:.2f}")
                        with col4:
                            st.metric("SMA50", f"₹{last['SMA50']:.2f}")
                        with col5:
                            st.metric("SMA200", f"₹{last['SMA200']:.2f}")

                        # Chart
                        chart = create_candlestick_chart(df.tail(90), ticker)
                        st.plotly_chart(chart, use_container_width=True)

                        # Trade notes
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            note_key = f"note_{stock}"
                            current_note = st.session_state.trade_notes.get(stock, "")
                            new_note = st.text_area(
                                "Trade Notes / Analysis",
                                value=current_note,
                                key=note_key,
                                placeholder="Enter your analysis, entry/exit levels, market conditions...",
                                height=100
                            )
                            st.session_state.trade_notes[stock] = new_note

                        with col2:
                            st.markdown("**Trade Decision:**")
                            decision = st.radio(
                                "Action",
                                ["⏳ Watching", "🟢 Buy", "🔴 Sell", "⏸️ Hold"],
                                key=f"decision_{stock}",
                                horizontal=False
                            )

                        # Quick levels
                        st.markdown("**Key Levels:**")
                        level_cols = st.columns(4)
                        with level_cols[0]:
                            st.text_input("Entry", key=f"entry_{stock}", placeholder="Entry price")
                        with level_cols[1]:
                            st.text_input("Stop Loss", key=f"sl_{stock}", placeholder="Stop loss")
                        with level_cols[2]:
                            st.text_input("Target 1", key=f"t1_{stock}", placeholder="Target 1")
                        with level_cols[3]:
                            st.text_input("Target 2", key=f"t2_{stock}", placeholder="Target 2")

                except Exception as e:
                    st.error(f"Error loading {stock}: {str(e)}")

            # Export watchlist
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Clear Watchlist"):
                    st.session_state.watchlist = []
                    st.session_state.trade_notes = {}
                    st.rerun()
            with col2:
                # Export notes
                if st.session_state.trade_notes:
                    notes_text = "WATCHLIST NOTES\\n" + "="*40 + "\\n"
                    for stock, note in st.session_state.trade_notes.items():
                        if note:
                            notes_text += f"\\n{stock}:\\n{note}\\n"
                    st.download_button("📥 Export Notes", notes_text, "watchlist_notes.txt")
            with col3:
                csv_data = ",".join(st.session_state.watchlist)
                st.download_button("📥 Export Tickers", csv_data, "watchlist.csv")

        else:
            st.info("👆 Select stocks from the dropdown above to add them to your watchlist")

            # Show some suggestions
            st.markdown("### Suggested Stocks to Watch")
            st.markdown("Based on recent screener results, consider adding:")

            # Quick scan for suggestions
            suggestions = []
            sample_stocks = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                           'BHARTIARTL.NS', 'TATASTEEL.NS', 'HINDALCO.NS', 'BAJFINANCE.NS', 'SBIN.NS']
            for ticker in sample_stocks:
                try:
                    df = fetch_ohlc_extended(ticker, period='6mo')
                    if df.empty:
                        continue
                    df = add_all_indicators(df)
                    signal = detect_swing_signal(df, min_score=35)
                    if signal.get('signal'):
                        suggestions.append({
                            'Stock': ticker.replace('.NS', ''),
                            'Grade': signal.get('signal_grade'),
                            'RSI': f"{signal.get('rsi', 0):.1f}"
                        })
                except:
                    continue

            if suggestions:
                st.dataframe(pd.DataFrame(suggestions), use_container_width=True, hide_index=True)
            else:
                st.write("No strong signals in top stocks currently. Add stocks manually to watch.")

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

    elif page == "📈 Backtest Strategy":
        st.header("Strategy Backtesting")

        col1, col2, col3 = st.columns(3)

        with col1:
            if nifty200:
                stock_options = [t.replace('.NS', '') for t in nifty200]
                selected = st.selectbox("Select Stock", stock_options, key='backtest_stock')
                ticker = f"{selected}.NS"
            else:
                ticker = st.text_input("Enter Ticker", "RELIANCE.NS", key='backtest_ticker')

        with col2:
            period = st.selectbox("Backtest Period", ['1y', '2y', '3y', '5y'], index=1)

        with col3:
            initial_capital = st.number_input("Initial Capital (₹)", 100000, 10000000, 500000, 50000)

        col1, col2 = st.columns(2)
        with col1:
            risk_per_trade = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
        with col2:
            min_score = st.slider("Min Signal Score", 30, 70, 50, key='backtest_score')

        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner(f"Backtesting {ticker} over {period}..."):
                results = backtest_strategy(
                    ticker, period=period,
                    initial_capital=initial_capital,
                    risk_per_trade=risk_per_trade,
                    min_score=min_score
                )

            if 'error' in results:
                st.error(results['error'])
                return

            if results.get('total_trades', 0) == 0:
                st.warning("No trades were generated during the backtest period.")
                return

            st.success("Backtest Complete!")

            # Performance Metrics
            st.subheader("📊 Performance Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                return_color = "normal" if results['total_return_pct'] >= 0 else "inverse"
                st.metric("Total Return", f"{results['total_return_pct']:.2f}%",
                          delta_color=return_color)
                st.metric("Final Capital", f"₹{results['final_capital']:,.0f}")

            with col2:
                st.metric("Total Trades", results['total_trades'])
                st.metric("Win Rate", f"{results['win_rate']:.1f}%")

            with col3:
                st.metric("Winning Trades", results['winning_trades'])
                st.metric("Losing Trades", results['losing_trades'])

            with col4:
                pf_display = f"{results['profit_factor']:.2f}" if results['profit_factor'] != float('inf') else "∞"
                st.metric("Profit Factor", pf_display)
                st.metric("Max Drawdown", f"{results['max_drawdown_pct']:.2f}%")

            st.markdown("---")

            # Additional Stats
            st.subheader("📈 Trade Statistics")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Avg Win", f"{results['avg_win_pct']:.2f}%")
            with col2:
                st.metric("Avg Loss", f"{results['avg_loss_pct']:.2f}%")
            with col3:
                st.metric("Avg Holding Days", f"{results['avg_holding_days']:.1f}")

            # Equity Curve
            st.subheader("📉 Equity Curve")
            if results.get('equity_curve'):
                equity_chart = create_equity_curve(
                    results['equity_curve'],
                    results['trades'],
                    ticker
                )
                st.plotly_chart(equity_chart, use_container_width=True)

            # Trade History
            st.subheader("📋 Trade History")
            if results.get('trades'):
                trades_df = pd.DataFrame(results['trades'])
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date']).dt.strftime('%Y-%m-%d')
                trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date']).dt.strftime('%Y-%m-%d')

                st.dataframe(
                    trades_df.style.format({
                        'entry_price': '₹{:.2f}',
                        'exit_price': '₹{:.2f}',
                        'pnl': '₹{:,.2f}',
                        'pnl_percent': '{:.2f}%'
                    }).applymap(
                        lambda x: 'color: #00ff00' if isinstance(x, (int, float)) and x > 0 else 'color: #ff6b6b' if isinstance(x, (int, float)) and x < 0 else '',
                        subset=['pnl', 'pnl_percent']
                    ),
                    use_container_width=True
                )

    elif page == "💰 Risk Calculator":
        st.header("Position Size & Risk Calculator")

        st.markdown("""
        Use this calculator to determine optimal position sizing based on your risk tolerance.
        **The 2% Rule**: Never risk more than 2% of your capital on a single trade.
        """)

        col1, col2 = st.columns(2)

        with col1:
            account_size = st.number_input("Account Size (₹)", 50000, 10000000, 500000, 10000)
            entry_price = st.number_input("Entry Price (₹)", 1.0, 100000.0, 100.0, 1.0)

        with col2:
            stop_loss = st.number_input("Stop Loss Price (₹)", 1.0, 100000.0, 95.0, 1.0)
            risk_percent = st.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.25) / 100

        if st.button("💰 Calculate Position", type="primary"):
            if stop_loss >= entry_price:
                st.error("Stop loss must be below entry price for long positions!")
            else:
                result = calculate_risk_metrics(account_size, entry_price, stop_loss, risk_percent)

                st.markdown("---")
                st.subheader("📊 Position Sizing Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Position Size", f"{result['position_size']} shares")
                    st.metric("Position Value", f"₹{result['position_value']:,.2f}")

                with col2:
                    st.metric("Risk Amount", f"₹{result['risk_amount']:,.2f}")
                    st.metric("Risk Per Share", f"₹{result['risk_per_share']:.2f}")

                with col3:
                    st.metric("Position % of Account", f"{result['position_percent']:.1f}%")
                    st.metric("Risk %", f"{result['risk_percent']:.1f}%")

                # Visual breakdown
                st.markdown("---")
                st.subheader("📈 Trade Visualization")

                # Simple profit/loss scenarios
                scenarios = pd.DataFrame({
                    'Scenario': ['Stop Loss Hit', 'Break Even', '+5% Gain', '+10% Gain', '+15% Gain'],
                    'Exit Price': [
                        stop_loss,
                        entry_price,
                        entry_price * 1.05,
                        entry_price * 1.10,
                        entry_price * 1.15
                    ],
                    'P&L': [
                        (stop_loss - entry_price) * result['position_size'],
                        0,
                        (entry_price * 0.05) * result['position_size'],
                        (entry_price * 0.10) * result['position_size'],
                        (entry_price * 0.15) * result['position_size']
                    ]
                })
                scenarios['Exit Price'] = scenarios['Exit Price'].apply(lambda x: f"₹{x:.2f}")
                scenarios['P&L'] = scenarios['P&L'].apply(lambda x: f"₹{x:,.2f}")

                st.table(scenarios)


if __name__ == '__main__':
    main()
