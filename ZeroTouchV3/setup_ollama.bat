@echo off
echo ==================================================
echo   ZeroTouch V3 Ollama Setup
echo ==================================================
echo This script will download the Llama 3.2 model for you.
echo Make sure you have Ollama installed first (from https://ollama.com/)
echo.
echo Press any key to start the download...
pause >nul

echo.
echo Pulling llama3.2 model...
ollama pull llama3.2

if %errorlevel% neq 0 (
    echo.
    echo ❌ Failed to pull the model. Please ensure Ollama is installed and running.
) else (
    echo.
    echo ✅ Model pulled successfully! You can now run ZeroTouch.exe
)
echo.
pause
