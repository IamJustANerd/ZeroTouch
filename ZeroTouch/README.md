# ZeroTouch: Operating Room Interface

ZeroTouch is a premium, AI-powered touchless interface designed for sterile environments like Operating Rooms (OR). It leverages computer vision for hand gesture tracking and natural language processing for audio assistance, allowing surgeons and medical staff to interact with digital systems without physical contact.

## Key Features
- **Touchless Gesture Control**: Real-time hand tracking and gesture classification via MediaPipe and TensorFlow.
- **Audio Assistance**: Integrated voice-command interface for hands-free queries and system control.
- **Glassmorphism UI**: High-performance, semi-transparent HUD overlay built with PyQt6.
- **Operational Modes**: Support for combined Camera/Audio HUD, Camera-only, or Audio-only assistance.

---

## Getting Started

### Prerequisites
- **Python 3.12** (Recommended for compatibility with MediaPipe/TensorFlow)
- Webcam (for gesture tracking)
- Microphone (for audio assistance)

### Installation & Setup

1. **Create a Virtual Environment**:
   ```powershell
   py -3.12 -m venv venv
   ```

2. **Activate the Virtual Environment**:
   - **PowerShell**: `.\venv\Scripts\Activate.ps1`
   - **Command Prompt**: `.\venv\Scripts\activate.bat`
   - **Git Bash**: `source venv/Scripts/activate`

3. **Install Dependencies**:
   ```powershell
   pip install opencv-python mediapipe tensorflow pyautogui numpy matplotlib PyQt6
   ```

4. **Download MediaPipe Model**:
   Download the hand landmarker task file to the project root:
   ```powershell
   curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```

---

## How to Run

### 1. Launching the App
To start the main launcher (which allows mode selection and settings configuration):
```powershell
python launcher_pyqt.py
```

### 2. Operational Modes
Once the launcher opens, you can select:
- **Camera & Audio Assistance**: Full HUD overlay with both vision and voice.
- **Camera Assist. Only**: Visual gesture tracking overlay.
- **Audio Assist. Only**: Voice-driven chat interface.

---

## Integration (Import & Run)

If you wish to import the core components into your own Python scripts:

### Import
```python
from overlay_pyqt import ZeroTouchOverlay
from audio_assist_ui import ZeroTouchAudioAssist
```

### Run (Example)
```python
import sys
from PyQt6.QtWidgets import QApplication

def run_overlay():
    app = QApplication(sys.argv)
    
    # Create the overlay (debug=True shows the camera feed)
    overlay = ZeroTouchOverlay(debug=True, show_chat=True)
    overlay.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    run_overlay()
```

---

## Project Structure
- `launcher_pyqt.py`: Main entry point and settings launcher.
- `hand_tracker.py`: Core hand tracking and landmark extraction logic.
- `overlay_pyqt.py`: The visual HUD overlay implementation.
- `audio_assist_ui.py`: The audio assistant chat interface.
- `documentation/`: Detailed guides on architecture, training, and setup.
