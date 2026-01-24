#!/usr/bin/env python3
"""
Nifty 200 Swing Trading App Launcher

This script launches the Swing Trading Dashboard application.
Simply run: python run_app.py

Features:
- Professional swing trading strategy with 6 technical indicators
- Real-time Nifty 200 stock screening
- Individual stock analysis with interactive charts
- Historical backtesting with performance metrics
- Position sizing and risk calculator
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check and install required dependencies."""
    required = [
        'streamlit',
        'yfinance',
        'pandas',
        'numpy',
        'plotly',
        'mplfinance',
        'matplotlib'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("Dependencies installed successfully!")

    return True

def main():
    """Launch the Swing Trading Dashboard."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, 'dashboard.py')

    # Check if dashboard exists
    if not os.path.exists(dashboard_path):
        print(f"Error: dashboard.py not found at {dashboard_path}")
        sys.exit(1)

    # Check dependencies
    print("Checking dependencies...")
    check_dependencies()

    print("\n" + "="*60)
    print("   NIFTY 200 SWING TRADING APP")
    print("   Professional Multi-Indicator Strategy")
    print("="*60)
    print("\nStarting the application...")
    print("The app will open in your default browser.\n")

    # Launch streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            dashboard_path,
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\nApp stopped by user.")
    except Exception as e:
        print(f"Error launching app: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
