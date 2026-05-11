# Audio Assist Mode
    
Operational mode focused on voice-to-text interaction and intelligent chat assistance.

## 1. Minimalist Chat HUD (`audio_assist_ui.py`)

The UI is inspired by modern web chatbots (GPT/Claude) but optimized for a floating OS overlay.

### AIChatInput Component
- **Aesthetic**: White glassmorphic container with 32px corner radius.
- **Dynamic Interaction**:
    - **Collapsed (68px)**: Standard input state.
    - **Expanded (128px)**: Reveals advanced controls ("Think", "Deep Search").
- **Animations**: Uses `QPropertyAnimation` for height transitions.
- **Placeholder Cycling**: Automatically rotates through medical and technical prompt ideas every 3s.

## 2. Controls & Toggles

| Control | Function |
|---------|----------|
| **💡 Think** | Enables reasoning-heavy model processing. |
| **🌐 Deep Search** | Activates web/database search integration. |
| **🎤 Mic** | Trigger for manual voice recording. |
| **📎 Attach** | Placeholder for diagnostic image/file attachment. |

## 3. Workflow Integration

1. Launch ZeroTouch.
2. Select **Audio Assist. Only**.
3. Use the floating chat input at the bottom of the screen.
4. Voice notes and OpenClaw responses are displayed in this context.
