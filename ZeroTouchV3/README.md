# ZeroTouch V3 (Voice Assistant App)

Welcome to ZeroTouch! This app allows you to control your Windows PC and interact with an AI completely hands-free using voice commands in Indonesian.

## Prerequisites

Before running the app, make sure you have the following installed on your Windows machine:
1. **Python 3.10 or 3.11** (Download from [python.org](https://www.python.org/downloads/)). Make sure to check the box **"Add Python to PATH"** during installation.
2. **Ollama** (Download from [ollama.com](https://ollama.com/)). This runs the AI model locally on your computer.

## Setup Instructions

### 1. Download the AI Model
After installing Ollama, you need to download the brain of the assistant (Llama 3.2). 
We have included a simple script to do this for you:
- Double-click the file named `setup_ollama.bat` inside this folder.
- A terminal will open and download the model. Wait for it to say **"Model pulled successfully!"**.

### 2. Install the Required Libraries
You need to install the dependencies required by the Python script. Open a terminal (Command Prompt or PowerShell) inside this `ZeroTouchV3` folder and run:

```powershell
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# Install the required libraries
pip install -r requirements.txt
```

*(Note: Downloading the requirements will take a while as it includes heavy libraries like PyTorch and LangChain).*

## How to Run

Whenever you want to start the app, open your terminal inside this folder and run:

```powershell
# Make sure your virtual environment is active
.\venv\Scripts\activate

# Run the app
python zerotouch_v3.py
```

## How to Use
1. The app will open a small transparent window in the bottom right corner of your screen.
2. Say **"Halo ZeroTouch"** to wake the assistant up.
3. Wait for the beep, then speak your command in Indonesian (e.g., *"Buka aplikasi Excel"* or *"Zoom layar"*).
4. The assistant will perform the action and reply to you out loud!

## Troubleshooting
- **"Model not found" or HuggingFace Connection Errors**: The app downloads a small transcription model on its first run. If your internet blocks HuggingFace, we have added a bypass in the code. If it still fails, ensure your internet connection doesn't block `huggingface.co`.
- **Microphone not picking up audio**: Ensure your microphone is set as the default recording device in Windows Sound Settings.
- **Port 5003 in use**: The app runs a local bridge on port 5003. Make sure you don't have multiple instances of the app running at the same time. If it freezes, press `Ctrl+C` in the terminal to forcefully kill it.
