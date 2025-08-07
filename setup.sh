#!/bin/bash
# setup.sh: Setup script for federated-learning-cifar10 project
# Usage: bash setup.sh

set -e

# Create virtual environment if not exists
echo "Creating Python virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
if [ -f requirements.txt ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found!"
    exit 1
fi

# Install project in editable mode
if [ -f setup.py ]; then
    echo "Installing project in editable mode..."
    pip install -e .
fi

echo "Setup complete! To activate the environment later, run: source .venv/bin/activate"
