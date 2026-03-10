@echo off
REM ======================================
REM  IDS Web App - Quick Start Script
REM ======================================

echo.
echo ======================================
echo   IDS - Intrusion Detection System
echo   Streamlit Web Application  
echo ======================================
echo.

REM Check if in correct directory
if not exist "app.py" (
    echo [ERROR] app.py not found!
    echo Please run this script from Do_An_Chuyen_Nganh directory
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python first
    pause
    exit /b 1
)

echo [INFO] Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Streamlit not installed
    echo Installing required packages...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo [OK] Dependencies ready
echo.
echo ===========================================
echo  Starting Streamlit app...
echo  App will open at: http://localhost:8501
echo ===========================================
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start Streamlit
streamlit run app.py

pause
