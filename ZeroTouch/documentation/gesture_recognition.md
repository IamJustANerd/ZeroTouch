# Gesture Recognition System (Universal)

ZeroTouch use universal gestures. No modes. Gestures work same in all apps.

## 1. 3-State Logic (Clutch)

Safety clutch prevent accidental move.

| State | Condition | Effect |
|-------|-----------|--------|
| **IDLE** | No hand | System locked. |
| **LISTENING** | Hold "Open Palm" | Progress bar fill. |
| **ACTIVE** | Hold "Open Palm" 2.0s | Gestures control OS. |

## 2. Universal Gestures (MLP)

| Gesture | OS Action | Detail |
|---------|-----------|--------|
| **Open Palm** | Move Mouse | High precision smoothing. |
| **Pointing** | Left Click | One-shot trigger. |
| **Fist** | Drag & Drop | Mouse down while held. |
| **Pinch** | Zoom Scroll | Proportional to finger distance delta. |

### Pinch Zoom
- Measure distance between Thumb tip (4) and Index tip (8).
- Delta (change) map to scroll strength.
- Big finger move -> fast zoom. Small move -> slow zoom.
- **Noise Filter**: Jumps > 5% distance are ignored to prevent glitches.
