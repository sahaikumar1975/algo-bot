"""
Main entry point - runs the comprehensive trading dashboard
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
    screen_weekly_breakout, calculate_sector_performance, get_sector,
    get_sector_momentum_score, get_swing_focus_sectors, get_top_stocks_by_sector,
    get_all_sectors
)

from dashboard import main as run_dashboard

# Run the dashboard
run_dashboard()

