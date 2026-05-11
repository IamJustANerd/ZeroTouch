# Project Overview: ZeroTouch

This document provides a high-level overview of the ZeroTouch project and serves as a guide to the project's documentation and entry points.

## Project Vision
ZeroTouch aims to bridge the gap between surgeons and technology in the sterile environment of an Operating Room. By removing the need for physical mice, keyboards, or touchscreens, we minimize contamination risks while maintaining high precision and responsiveness.

## Core Components

### 1. Vision System (`hand_tracker.py`, `gesture_classifier.keras`)
Uses MediaPipe to extract 21 hand landmarks in 3D space. These landmarks are then fed into a TensorFlow-based classifier to identify specific gestures (e.g., Pointing, Pinching, Clenching).

### 2. UI Overlay (`overlay_pyqt.py`, `launcher_pyqt.py`)
Built with **PyQt6**, the UI follows a "Glassmorphism" aesthetic—sleek, semi-transparent, and non-intrusive. It provides visual feedback for gesture detection and AI health status.

### 3. Audio Intelligence (`audio_assist_ui.py`)
A dedicated assistant that provides voice-driven interaction, allowing users to query system states or perform actions using natural language.

## Repository Layout
| File/Folder | Description |
| :--- | :--- |
| `README.md` | General setup and quick-start guide. |
| `documentation/` | In-depth technical documentation for each subsystem. |
| `venv/` | Isolated Python environment containing all necessary dependencies. |
| `hand_landmarker.task` | Pre-trained MediaPipe model for hand landmark detection. |

## Documentation Guide
For more specific details, refer to:
- [Setup & Dependencies](setup.md)
- [Gesture Recognition Logic](gesture_recognition.md)
- [UI Launcher Design](ui_launcher.md)
- [Executable Packaging Guide](executable_guide.md)

---
*Created as part of the project documentation requirements.*
