#!/bin/bash

# Create a virtual environment named find_your_state
python3 -m venv find_your_state

# Activate the virtual environment
source find_your_state/bin/activate

# Check Python version
which python

# Upgrade pip
python -m pip install --upgrade pip

# Install packages from requirements.txt
pip install -r requirements.txt