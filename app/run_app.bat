@echo off
REM Run Streamlit App Locally
REM This script runs the Coconut Price Predictor without Docker

echo.
echo ========================================
echo Coconut Auction Price Predictor
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking Python installation...
python --version

REM Check if virtual environment exists
if not exist "venv" (
    echo.
    echo [2/3] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [2/3] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are already installed
echo.
echo [3/3] Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies - first time setup...
    echo This may take a few minutes...
    pip install -q -r deployment_requirements.txt
    if errorlevel 1 goto :install_error
    echo Dependencies installed successfully!
) else (
    echo Dependencies already installed, skipping installation.
)

REM Check if model artifacts exist
if not exist "..\model\outputs\model.joblib" (
    echo.
    echo ========================================
    echo WARNING: Model artifacts not found!
    echo ========================================
    echo.
    echo Model artifacts are missing in the model/outputs/ folder.
    echo.
    echo Please run the training script first:
    echo   cd ..\model
    echo   python train_model.py
    echo.
    echo This will generate:
    echo   - model/outputs/model.joblib
    echo   - model/outputs/scaler.joblib
    echo   - model/outputs/feature_names.joblib
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting Streamlit Application...
echo ========================================
echo.
echo The app will open in your default browser
echo If not, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo ========================================
echo.

REM Run Streamlit app
streamlit run app.py

REM If Streamlit exits with error
if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo Check the error messages above
    pause
)
goto :eof

:install_error
echo.
echo ERROR: Failed to install dependencies
pause
exit /b 1
