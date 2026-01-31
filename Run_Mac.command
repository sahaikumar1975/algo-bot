#!/bin/bash
echo "🚀 Starting SMA2150 Command Center..."

# Ensure we are in the script's directory (Resolving Symlinks)
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR=$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )
cd "$DIR"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 could not be found."
    echo "Please install Python 3."
    exit 1
fi

# Run Streamlit App
streamlit run app.py
