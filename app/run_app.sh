#!/bin/bash
# Run Streamlit App Locally
# This script runs the Coconut Price Predictor without Docker

echo ""
echo "========================================"
echo "Coconut Auction Price Predictor"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

echo "[1/3] Checking Python installation..."
python3 --version

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "[2/3] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
else
    echo "[2/3] Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are already installed
echo ""
echo "[3/3] Checking dependencies..."
python -c "import streamlit" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies (first time setup)..."
    echo "This may take a few minutes..."
    pip install -q -r deployment_requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
    echo "Dependencies installed successfully!"
else
    echo "Dependencies already installed, skipping installation."
fi

# Check if model artifacts exist
if [ ! -f "../model/outputs/model.joblib" ]; then
    echo ""
    echo "========================================"
    echo "WARNING: Model artifacts not found!"
    echo "========================================"
    echo ""
    echo "Model artifacts are missing in the model/outputs/ folder."
    echo ""
    echo "Please run the training script first:"
    echo "  cd ../model"
    echo "  python train_model.py"
    echo ""
    echo "This will generate:"
    echo "  - model/outputs/model.joblib"
    echo "  - model/outputs/scaler.joblib"
    echo "  - model/outputs/feature_names.joblib"
    echo ""
    exit 1
fi

echo ""
echo "========================================"
echo "Starting Streamlit Application..."
echo "========================================"
echo ""
echo "The app will open in your default browser"
echo "If not, go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo "========================================"
echo ""

# Run Streamlit app
streamlit run app.py

# If Streamlit exits with error
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Application failed to start"
    echo "Check the error messages above"
fi
