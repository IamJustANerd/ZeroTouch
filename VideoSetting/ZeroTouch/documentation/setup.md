# Project Setup - ZeroTouch

This document describes the environment setup process for the ZeroTouch project.

## 1. Virtual Environment (venv)
To ensure compatibility with MediaPipe and TensorFlow, please use **Python 3.12**.

### Creation
```powershell
py -3.12 -m venv venv
```

### Activation
- **PowerShell**: `.\venv\Scripts\Activate.ps1`
- **Cmd**: `.\venv\Scripts\activate.bat`
- **Git Bash**: `source venv/Scripts/activate`

---

## 2. Dependencies
Install the core libraries for vision processing and OS emulation:

```powershell
pip install opencv-python mediapipe tensorflow pyautogui numpy matplotlib
```

| Library | Purpose |
| :--- | :--- |
| **OpenCV** | Webcam and image processing. |
| **MediaPipe** | Hand landmark extraction (Tasks API). |
| **TensorFlow** | Gesture classification models. |
| **PyAutoGUI** | OS mouse/keyboard emulation. |

## 3. External Assets
The MediaPipe Tasks API requires a model file:
```bash
curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```
