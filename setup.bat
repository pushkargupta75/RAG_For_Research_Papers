@echo off
REM sci_synth Setup Script for Windows
REM Automates environment setup and dependency installation

echo ========================================
echo sci_synth Setup Script
echo ========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python 3.11+
    exit /b 1
)
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Skipping...
) else (
    python -m venv venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
echo.

REM Verify installation
echo Verifying installation...
python -c "import streamlit; import langchain; import chromadb; print('Core packages installed successfully')"
if %errorlevel% neq 0 (
    echo Error: Package installation verification failed
    exit /b 1
)
echo.

REM Create directories
echo Creating directories...
if not exist "data" mkdir data
if not exist "chroma_db" mkdir chroma_db
if not exist "notebooks" mkdir notebooks
echo Directories created
echo.

REM Copy .env.example to .env if not exists
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo .env file created
    echo WARNING: Please edit .env and add your API keys
) else (
    echo .env file already exists. Skipping...
)
echo.

REM Summary
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file and add your API keys
echo 2. Place PDF papers in data\ directory
echo 3. Run Streamlit app:
echo    streamlit run src\app.py
echo.
echo Or explore the Jupyter notebooks:
echo    jupyter notebook notebooks\
echo.
echo For more information, see README.md
echo.
pause
