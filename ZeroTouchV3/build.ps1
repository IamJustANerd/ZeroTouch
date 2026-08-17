$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\micha\OneDrive\Documents\Projects\ZeroTouch"
$ZeroTouchV3 = "$ProjectRoot\zerotouchv3"
$ModelsDir = "$ProjectRoot\VoiceSetting\zerotouch_voicerecognition\models"
$DataDir = "$ZeroTouchV3\data"
$PiperExe = "$ProjectRoot\venv\Scripts\piper.exe"
$PyInstaller = "$ProjectRoot\venv\Scripts\pyinstaller.exe"

Write-Host "Building ZeroTouchV3..." -ForegroundColor Cyan

cd $ZeroTouchV3

& $PyInstaller `
    --noconfirm `
    --onedir `
    --name "ZeroTouch" `
    --add-data "$ModelsDir;models" `
    --add-data "$DataDir;data" `
    --add-data "$PiperExe;." `
    --hidden-import "langchain" `
    --hidden-import "langchain_core" `
    --hidden-import "langchain_ollama" `
    --hidden-import "langgraph" `
    --hidden-import "openwakeword" `
    --hidden-import "faster_whisper" `
    --hidden-import "PyQt6" `
    --hidden-import "ctranslate2" `
    zerotouch_v3.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build Complete! Executable is located in dist\ZeroTouch\" -ForegroundColor Green
} else {
    Write-Host "Build failed." -ForegroundColor Red
}
