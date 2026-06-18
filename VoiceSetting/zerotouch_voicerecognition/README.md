# ZeroTouch Voice Recognition (ProTel)

**ZeroTouch** is a next-generation, hands-free medical interface. It empowers healthcare professionals to securely and hygienically control clinical systems entirely through Indonesian voice commands. By eliminating the need to physically touch computer hardware, ZeroTouch ensures a sterile environment while granting doctors full control to:
- Access Electronic Medical Records (EMR)
- Retrieve and view DICOM images via PACS (Picture Archiving and Communication System)
- Navigate and manipulate 3D medical objects

Built as a completely offline, edge-AI telematics solution, the system seamlessly integrates wake word detection, real-time transcription, and automated hospital software execution.

### 👨‍💻 My Role: STT & TTS Lead Engineer
I served as the Lead Engineer for the **Speech-to-Text (STT)** and **Text-to-Speech (TTS)** modules—essentially building the "ears" and "mouth" of the entire ZeroTouch ecosystem. My core responsibilities included:
- **Low-Latency Transcription:** Architecting a highly responsive STT pipeline using `faster-whisper`, specifically optimized for complex Indonesian medical terminology.
- **Custom Wake Word AI:** Training and deploying a lightweight `openwakeword` detection model to ensure idle computational efficiency and offline readiness.
- **Natural Voice Feedback:** Configuring Piper TTS with a custom phonetic pre-processing layer to synthesize natural, real-time voice responses.

The communication layer I developed serves as the critical bridge for the system: capturing doctors' spoken commands, forwarding them to a Local LLM (Ollama) for reasoning, and passing the intent to OpenClaw (developed by my teammates) for physical execution.

## ✨ Features & Tech Stack

- **Custom Wake Word Detection (`openwakeword`)**
  - Train and deploy your own wake words (e.g., "Hello Zero Touch").
  - Includes a robust data-preparation and training pipeline avoiding data leakage.
- **Real-Time Speech-to-Text (`faster-whisper`)**
  - Fast, accurate, offline transcription triggered by hands-free activation.
  - State-machine architecture that auto-pauses and saves transcriptions to JSON after silence.
- **Batch STT Processing**
  - Transcribe existing audio/video files effortlessly.
- **Text-to-Speech (`piper-tts`)**
  - High-quality, offline Indonesian TTS (`id_ID-news_tts-medium.onnx`).

## 📁 Project Structure

- `stt_activation.py` - Live, real-time STT pipeline listening for the custom "Hello Zero Touch" wake word.
- `train_wakeword_v2.py` & `prepare_data_v2.py` - Custom Wake Word training pipeline scripts.
- `stt.py` - Batch STT transcription for pre-recorded audio.
- `tts.py` / `tts_piper.py` - Text-to-Speech synthesis scripts.
- `models/` - Directory for storing local ONNX models (Wake word models and Piper TTS models). See `models/README.md` for specific model documentation.
- `text-from-stt/` - JSON outputs containing timestamps and transcriptions from live STT sessions.
- `training_data/` - Raw audio used to train the custom wake words.

## 🚀 Setup & Installation

It is recommended to run this project inside a Python virtual environment to manage dependencies securely.

1. **Create a virtual environment:**
   ```powershell
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

## 💻 Usage

Ensure your virtual environment is active before running any scripts.

### 1. Live Speech-to-Text (Wake Word Mode)
This script continuously listens to your microphone. When you say the wake word ("Hello Zero Touch"), it starts transcribing your speech until it detects silence.

```powershell
python stt_activation.py
```
*Transcriptions are saved in the `text-from-stt/` folder as JSON.*

### 2. Custom Wake Word Training
To train your own wake word or retrain the existing model on new data, use the V2 pipeline:

```powershell
# 1. Prepare, clean, and augment the data strictly
python prepare_data_v2.py

# 2. Extract features, run cross-validation, and export ONNX
python train_wakeword_v2.py
```
*Models are exported to the `models/` directory.*

### 3. Batch Speech-to-Text
Transcribe a specific audio file.

```powershell
python stt.py
```

### 4. Text-to-Speech (Indonesian)
Synthesize Indonesian text into audio.

```powershell
python tts_piper.py
```
*Outputs are saved in the `tts-audio/` folder.*

---
**Note on Git & Large Files:**
Large model files (`.onnx`) and massive augmented datasets (`training_data_v2/`, `*.npy`) should not be pushed to GitHub due to storage limits. They are included in `.gitignore` by default. Only push raw training data, scripts, and finalized `.onnx` models.
