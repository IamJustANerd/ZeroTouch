# Gesture Control and State Management

This document describes the hand gestures available in the Protel project and how they map to Windows OS controls via the `hand_tracker.py` script.

## 1. System States (Clutch System)
The system uses a `ClutchManager` to prevent accidental OS interactions.

| State | Description | Transition Logic |
| :--- | :--- | :--- |
| **IDLE** | System is locked. No OS control. | Detects "Open Palm" to enter **LISTENING**. |
| **LISTENING** | Preparing to activate. | Must hold "Open Palm" for **2.0 seconds** to enter **ACTIVE**. |
| **ACTIVE** | Gestures actively control the mouse. | Reverts to **IDLE** if hand is lost for > **1.0 second**. |

## 2. Gesture Mappings
When the state is **ACTIVE**, the following gestures are mapped to Windows OS actions:

### Open Palm
- **Action**: Mouse Movement + Mouse Up.
- **Behavior**: The cursor follows the hand (Middle Finger MCP center). Any active mouse press (from a Fist gesture) is released.

### Pointing
- **Action**: Single Left Click.
- **Behavior**: Performs a "one-shot" click. To click again, the user must briefly change gestures or move the hand.

### Fist
- **Action**: Mouse Down + Relative Mouse Movement (Drag).
- **Behavior**: Simulates holding the left mouse button down. Uses raw relative mouse motion (`MOUSEEVENTF_MOVE` via `ctypes`) to compute delta distances. This ensures seamless panning in applications like photo viewers and 3D space that ignore absolute cursor positions.

### Pinch
- **Action**: None (Detected).
- **Behavior**: The model recognizes this gesture, but it is currently a placeholder and does not trigger an OS action.

## 3. Technical Implementation
- **Library**: `pyautogui` for OS control, `mediapipe` for hand tracking, `tensorflow` (TFLite) for gesture classification.
- **Smoothing**: Uses an Exponential Moving Average (EMA) with an alpha of `0.4` to reduce jitter.
- **Fail-safe**: `pyautogui.FAILSAFE` is enabled. Moving the mouse to any corner of the screen will abort the script.
