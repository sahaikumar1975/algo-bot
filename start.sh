#!/bin/bash
# Nifty 200 Swing Trading App - Quick Start Script

echo "=============================================="
echo "   NIFTY 200 SWING TRADING APP"
echo "   Professional Multi-Indicator Strategy"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

# Install requirements if needed
echo "Checking dependencies..."
pip3 install -q -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Starting the app..."
echo "Open http://localhost:8501 in your browser"
echo ""

# Run the dashboard
streamlit run "$SCRIPT_DIR/dashboard.py" --server.headless true --browser.gatherUsageStats false
