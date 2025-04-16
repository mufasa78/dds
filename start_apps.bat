@echo off
echo ===================================================
echo Multilingual Deepfake Detection System Launcher
echo ===================================================
echo.

REM Check if venv_new exists
if not exist venv_new (
    echo Virtual environment not found. Creating a new one...
    python -m venv venv_new
    call venv_new\Scripts\activate
    echo Installing required packages...
    python -m pip install -r requirements.txt
) else (
    echo Activating virtual environment...
    call venv_new\Scripts\activate
)

REM Check dependencies
echo Checking dependencies...
python check_dependencies.py

echo.
echo Choose an application to run:
echo 1. Flask Web Application (port 5001)
echo 2. Streamlit Application (port 5000)
echo 3. Run both applications
echo 4. Test models and dependencies
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting Flask application...
    start cmd /k "call venv_new\Scripts\activate && python flask_app.py"
) else if "%choice%"=="2" (
    echo.
    echo Starting Streamlit application...
    start cmd /k "call venv_new\Scripts\activate && streamlit run app.py"
) else if "%choice%"=="3" (
    echo.
    echo Starting both applications...
    start cmd /k "call venv_new\Scripts\activate && python flask_app.py"
    timeout /t 2 > nul
    start cmd /k "call venv_new\Scripts\activate && streamlit run app.py"
) else if "%choice%"=="4" (
    echo.
    echo Testing models and dependencies...
    call venv_new\Scripts\activate && python test_models.py
    echo.
    pause
) else if "%choice%"=="5" (
    echo Exiting...
    exit
) else (
    echo Invalid choice. Please run the script again.
)

echo.
echo Applications started. Press any key to exit this window...
pause > nul
