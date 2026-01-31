#!/bin/bash
echo "🚀 Starting SMA2150 Command Center..."

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 could not be found."
    echo "Please install Python 3."
    exit 1
fi

# Run Streamlit App
streamlit run app.py
