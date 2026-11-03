@echo off
cd /d "%~dp0"
py -3 -c "import streamlit, google.genai, PIL, fpdf" 2>nul
if errorlevel 1 (
    echo Required packages are missing. Install them with:
    echo   py -3 -m pip install -r "%~dp0requirements.txt"
    pause
    exit /b 1
)
py -3 -m streamlit run "%~dp0app.py"
pause