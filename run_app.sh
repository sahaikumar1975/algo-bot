#!/bin/bash
echo "🚀 Starting SMA2150 Command Center..."

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Run Streamlit App
streamlit run app.py
