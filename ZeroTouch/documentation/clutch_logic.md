# Clutch Logic (State Management) - ZeroTouch

This document describes the 3-state system that ensures the ZeroTouch interface only translates gestures to OS commands when intentional.

## States
1.  **`IDLE`**: The hand tracker is running, but no commands are sent to the OS.
2.  **`LISTENING`**: Triggered by an **Open Palm**. Shows a 2-second **Loading Bar**.
3.  **`ACTIVE`**: Triggered after the 2-second countdown. Navigation is enabled.

## Interactions (Active State)
| Gesture | OS Action (via PyAutoGUI) |
| :--- | :--- |
| **Pointing (3)** | Move Mouse (Scaled to Screen). |
| **Fist (1)** | Left-Click Down (For dragging/rotating). |
| **Pinch (2)** | Left-Click (Selection). |

## Safety Controls
- **`r` Key (Reset)**: Force the state back to `LISTENING`.
- **Hand Loss**: When the hand leaves the frame, the system returns to `IDLE` after a **1.0-second grace period**.
- **Pinch Debounce**: One pinch = one click. Must release and re-pinch for another click.

## Mouse Smoothing (EMA Filter)
An **Exponential Moving Average** filter is applied to eliminate cursor jitter:
```
smoothed_position = α × raw_position + (1 - α) × previous_position
```
- **`α = 0.4`** (default): Good balance between responsiveness and stability.
- **Tuning**: Change `self.alpha` in `HandTracker.__init__()` (line ~95 of `hand_tracker.py`).
  - Lower values (0.2) = smoother but sluggish.
  - Higher values (0.6) = responsive but jittery.

## UI Elements
- **Loading Bar**: A progress bar (Red -> Green) that fills during the `LISTENING` phase.
- **State Label**: A clear text display showing either "IDLE", "LISTENING", or "ACTIVE".
