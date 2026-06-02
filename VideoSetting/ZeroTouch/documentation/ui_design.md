# UI Design: ZeroTouch Ecosystem

Unified theme for Launcher and Camera Assist HUD.

## 1. Design Tokens

| Token | Value | Description |
|-------|-------|-------------|
| **Primary** | `#00FFFF` | Cyan accent for active states. |
| **Danger** | `#FF4B4B` | Red for errors/close. |
| **Glass BG** | `rgba(30, 30, 30, 220)` | Translucent dark grey. |
| **Radius** | `24px` | Consistent rounded corners. |
| **Font** | `Inter, sans-serif` | Modern technical typeface. |

## 2. Components

### Launcher (Menu)
- Centered layout.
- Vertical gradient background.
- Floating translucent settings wheel.

### Camera Assist (HUD)
- Fixed bottom-right overlay.
- **Glassmorphism**: 220-opacity background with 40-opacity white border.
- **Dynamic Glow**: Border turn Cyan when system is `ACTIVE`.
- **Progress Bar**: Orange fill during `LISTENING` phase.

## 3. Visual States

- **IDLE**: Grey text, standard glass border.
- **LISTENING**: Orange accents, active progress bar.
- **ACTIVE**: Cyan accents, glowing cyan border.

## 4. Window Management

- **Always on Top**: The HUD uses `WindowStaysOnTopHint`.
- **Taskbar**: Visible in taskbar for easy focus retrieval.
- **Persistence**: A `QTimer` triggers `raise_()` every 1000ms to ensure the HUD remains above windows opened after the engine starts.
