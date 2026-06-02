"""
zerotouch_v0.py  –  Zero Touch v0
===================================
Unified launcher that merges:
  • Hand gesture recognition  (ZeroTouch/hand_tracker.py)
  • Live speech-to-text       (zerotouch_voicerecognition/stt_live.py)

States
------
  SLEEP  →  Red border.   Gesture + mic both inactive for cursor/command use.
  WAKE   →  Green border. Gesture cursor active. STT listens and sends to OpenClaw.

Wake triggers (either one works):
  • Hold Open-Palm gesture for 2 s  (ClutchManager reaches ACTIVE)
  • Say the wake word  "hey jarvis"

Sleep triggers (either one works):
  • Show Fist gesture for 2 s  while in WAKE state
  • Say "sleep" / "tidur" / "stop listening"  after wake word detected

STT → OpenClaw pipeline:
  Transcribed text  →  POST http://localhost:18789/v1/messages  (OpenClaw HTTP API)

Usage
-----
  python zerotouch_v0.py

Requirements (pip install):
  PyQt6, opencv-python, mediapipe, tensorflow, pyautogui,
  faster-whisper, openwakeword, sounddevice, numpy, requests
"""

import sys
import os
import time
import threading
import queue
import json
import math
import logging
import warnings

# ── Fix terminal encoding on Windows ─────────────────────────────
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import requests

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QFrame, QScrollArea, QPushButton,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint
from PyQt6.QtGui import (
    QColor, QPainter, QPen, QFont, QLinearGradient, QImage, QPixmap
)

# ────────────────────────────────────────────────────────────────
#  PATHS  – resolve sibling folders regardless of cwd
# ────────────────────────────────────────────────────────────────
ROOT          = os.path.dirname(os.path.abspath(__file__))
ZEROTOUCH_DIR = os.path.join(ROOT, "ZeroTouch")
STT_DIR       = os.path.join(ROOT, "VoiceSetting", "zerotouch_voicerecognition")

sys.path.insert(0, ZEROTOUCH_DIR)
sys.path.insert(0, STT_DIR)

# ────────────────────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────────────────────
OPENCLAW_BASE   = "http://localhost:18789"
OPENCLAW_TOKEN  = "1542658497515794168875165986594"

# STT
STT_MODEL_SIZE  = "small"
STT_DEVICE      = "cpu"
STT_COMPUTE     = "int8"
STT_THREADS     = 8
STT_LANGUAGE    = "id"          # Indonesian for voice recognition
WAKE_WORD_MODEL = "hey_jarvis"
WAKE_WORD_LABEL = "hey jarvis"
WAKE_WORD_THRESH = 0.9          # Wake word sensitivity
SAMPLE_RATE     = 16000
OWW_CHUNK       = 1280
AUDIO_GAIN      = 1.0           # No gain (mic is healthy, gain causes clipping)
SILENCE_THRESHOLD = 0.01        # Silence threshold for timeout
SILENCE_TIMEOUT = 3.5           # Silence timeout in seconds

# Gesture sleep trigger: hold Fist for N seconds
SLEEP_GESTURE_HOLD = 2.0

# Sleep words the STT may detect (lowercased substring match)
SLEEP_WORDS = ["sleep", "tidur", "stop listening", "berhenti", "diam"]


HOST_BRIDGE_URL = "http://localhost:5000/run"

# Keyword → app mapping (matches what host_bridge.py allows)
# Includes Indonesian speech patterns + common Whisper mishearing variants
APP_KEYWORDS = {
    # Microsoft Word
    "microsoft word": "word",
    "microsoft world": "word",  # Whisper mishearing
    "word":           "word",
    "world":          "word",   # Whisper mishearing in Indonesian mode
    "winword":        "word",
    "buka word":      "word",   # Indonesian: "open word"
    "buka microsoft": "word",
    "open word":      "word",
    "open microsoft": "word",

    # WhatsApp
    "whatsapp":       "whatsapp",
    "buka whatsapp":  "whatsapp",
    "open whatsapp":  "whatsapp",
    "watsap":         "whatsapp",  # Whisper phonetic variant

    # Notepad
    "notepad":        "notepad",
    "note pad":       "notepad",
    "buka notepad":   "notepad",
    "open notepad":   "notepad",
    "buka catatan":   "notepad",   # Indonesian: "open notes"
}

def send_to_openclaw(text: str) -> bool:
    """
    Parse the voice command for a known app name and call host_bridge.py directly.
    Falls back to printing if no app keyword is matched.
    """
    text_lower = text.lower()

    # Try to find a matching app keyword
    matched_app = None
    for keyword, app_key in APP_KEYWORDS.items():
        if keyword in text_lower:
            matched_app = app_key
            break

    if matched_app:
        try:
            r = requests.post(HOST_BRIDGE_URL, json={"action": matched_app}, timeout=5)
            if r.ok:
                print(f"[bridge] Launched '{matched_app}' via host_bridge.")
                return True
            else:
                print(f"[bridge] host_bridge error {r.status_code}: {r.text}")
                return False
        except Exception as e:
            print(f"[bridge] Failed to reach host_bridge: {e}")
            return False
    else:
        # No app keyword matched — voice command not supported yet
        print(f"[bridge] No app keyword matched in: {text!r}")
        print(f"[bridge] Supported: {list(set(APP_KEYWORDS.values()))}")
        return True  # Return True so the UI says 'Sent' not 'unreachable'



# ════════════════════════════════════════════════════════════════
#  GESTURE THREAD  (wraps HandTrackerEngine)
# ════════════════════════════════════════════════════════════════
class GestureThread(QThread):
    """Runs hand tracking in a background thread, emits signals to the UI."""
    frame_ready    = pyqtSignal(np.ndarray, str, float, str, float)  # img,state,prog,gest,fps
    wake_requested = pyqtSignal()
    sleep_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = False
        self._fist_hold_start = None  # for sleep gesture detection

    def run(self):
        try:
            from hand_tracker import HandTrackerEngine
        except ImportError as e:
            print(f"[gesture] Import error: {e}")
            return

        self._running = True
        self._engine  = HandTrackerEngine(debug=True, callback=self._cb)
        self._engine.run()

    def _cb(self, img, state, progress, mlp_gest, fps):
        self.frame_ready.emit(img, state, progress, mlp_gest, fps)

        # Wake: gesture ClutchManager reached ACTIVE
        if state == "ACTIVE":
            self.wake_requested.emit()

        # Sleep gesture: hold Fist for SLEEP_GESTURE_HOLD seconds
        if mlp_gest == "Fist":
            if self._fist_hold_start is None:
                self._fist_hold_start = time.time()
            elif time.time() - self._fist_hold_start >= SLEEP_GESTURE_HOLD:
                self.sleep_requested.emit()
                self._fist_hold_start = None
        else:
            self._fist_hold_start = None

    def set_active(self, active: bool):
        if hasattr(self, "_engine"):
            if active:
                self._engine.clutch.state = 'ACTIVE'
                self._engine.clutch.grace_period = 999999.0  # Lock hand tracker to remain active
            else:
                self._engine.clutch.state = 'IDLE'
                self._engine.clutch.progress = 0
                self._engine.clutch.grace_period = 1.0       # Reset default timeout

    def stop(self):
        self._running = False
        if hasattr(self, "_engine"):
            self._engine.running = False
        self.wait(3000)


# ════════════════════════════════════════════════════════════════
#  STT THREAD  (openwakeword + faster-whisper)
# ════════════════════════════════════════════════════════════════
class STTThread(QThread):
    wake_detected       = pyqtSignal()
    transcription_ready = pyqtSignal(str)
    status_changed      = pyqtSignal(str)   # "IDLE" | "LISTENING" | "THINKING"
    mic_volume          = pyqtSignal(float) # Live mic RMS volume

    def __init__(self):
        super().__init__()
        self._running      = True
        self._force_listen = False
        self._ptt_active   = threading.Event()  # Push-To-Talk active flag
        self._ptt_stop     = threading.Event()  # Signal to stop recording NOW

    def set_force_listen(self, force: bool):
        self._force_listen = force

    def ptt_start(self):
        """Call when Space is pressed – immediately start recording."""
        self._ptt_active.set()
        self._ptt_stop.clear()

    def ptt_stop(self):
        """Call when Space is released – stop recording and transcribe."""
        self._ptt_stop.set()
        self._ptt_active.clear()

    def run(self):
        try:
            import sounddevice as sd
            from faster_whisper import WhisperModel
            from openwakeword.model import Model as WakeWordModel
        except ImportError as e:
            print(f"[stt] Import error: {e}  – STT disabled.")
            return

        print("[stt] Loading Whisper model…")
        stt = WhisperModel(STT_MODEL_SIZE, device=STT_DEVICE,
                           compute_type=STT_COMPUTE, cpu_threads=STT_THREADS)
        print("[stt] Loading wake-word model…")
        oww = WakeWordModel(wakeword_models=[WAKE_WORD_MODEL],
                            inference_framework="onnx")
        print("[stt] Ready.")

        from enum import Enum, auto
        class S(Enum):
            IDLE      = auto()
            LISTENING = auto()

        audio_q: queue.Queue = queue.Queue()

        def _cb(indata, frames, t, status):
            audio_q.put(indata.copy())

        state         = S.IDLE
        buf           = []
        last_speech   = 0.0
        bg_rms        = 0.001
        session_max_rms = 0.0

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=OWW_CHUNK,
                                callback=_cb)

        self.status_changed.emit("IDLE")

        with stream:
            while self._running:
                try:
                    block = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                mono = block[:, 0]
                rms_raw = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
                
                # Speech is detected if the current volume is at least 2.0x the background noise
                # (plus a tiny absolute floor to prevent triggering in total silence)
                is_speech = rms_raw > (bg_rms * 2.0 + 0.001)

                # Send raw unamplified RMS to the UI so the visual bar is honest
                self.mic_volume.emit(rms_raw)

                # Boost audio strictly for the AI models (openwakeword & whisper)
                mono_boosted = np.clip(mono * AUDIO_GAIN, -1.0, 1.0)

                if state == S.IDLE:
                    # Check for Wake Word using openwakeword if not already forced or PTT
                    # Convert mono to int16 for openwakeword
                    chunk_i16 = np.clip(mono * 32767, -32768, 32767).astype(np.int16)
                    predictions = oww.predict(chunk_i16)
                    
                    wake_triggered = False
                    for ww_name, score in predictions.items():
                        if score >= WAKE_WORD_THRESH:
                            wake_triggered = True
                            self.wake_detected.emit()
                            break

                    if self._ptt_active.is_set() or self._force_listen or wake_triggered:
                        state = S.LISTENING
                        buf = [mono_boosted.copy()]
                        last_speech = time.time()
                        session_max_rms = rms_raw
                        self.status_changed.emit("LISTENING")
                    else:
                        # Dynamic Background Noise Tracking (only update when NOT speaking)
                        bg_rms = 0.95 * bg_rms + 0.05 * rms_raw

                elif state == S.LISTENING:
                    buf.append(mono_boosted.copy())
                    
                    if rms_raw > session_max_rms:
                        session_max_rms = rms_raw
                        
                    # Reset silence timer if rms is above the silence threshold
                    if rms_raw >= SILENCE_THRESHOLD:
                        last_speech = time.time()

                    # PTT released → transcribe immediately.
                    ptt_released = self._ptt_stop.is_set() and not self._ptt_active.is_set()
                    
                    # Silence timeout reached (only if not holding PTT)
                    silence_dur = time.time() - last_speech
                    silence_timeout_reached = silence_dur >= SILENCE_TIMEOUT and not self._ptt_active.is_set()

                    if ptt_released or silence_timeout_reached:
                        self.status_changed.emit("THINKING")
                        audio_data = np.concatenate(buf)
                        
                        # Use updated initial_prompt
                        segs, _ = stt.transcribe(
                            audio_data, beam_size=5, language=STT_LANGUAGE,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=400),
                            condition_on_previous_text=True,
                            initial_prompt="Berikut adalah percakapan dalam bahasa Indonesia, kata yang dibicarakan adalah kata formal dan informal, pastikan hasil transkripsi adalah kata yang valid.",
                        )
                        text = " ".join(s.text.strip() for s in segs).strip()
                        print(f"[stt] Transcribed: {text!r}")
                        # Always emit so overlay ALWAYS shows what was (or wasn't) heard
                        self.transcription_ready.emit(text if text else "(nothing heard)")
                        
                        # Reset states
                        buf   = []
                        state = S.IDLE
                        self._force_listen = False
                        self._ptt_stop.clear()   # Reset PTT stop flag
                        session_max_rms = 0.0
                        
                        # Fixes: Reset oww model and clear audio queue to prevent leftover audio
                        if hasattr(oww, 'reset'):
                            oww.reset()
                        with audio_q.mutex:
                            audio_q.queue.clear()
                            
                        self.status_changed.emit("IDLE")

    def stop(self):
        self._running = False
        self.wait(5000)


# ════════════════════════════════════════════════════════════════
#  STT FLOATING OVERLAY WINDOW
# ════════════════════════════════════════════════════════════════
class STTOverlayWindow(QWidget):
    """Permanent floating window to clearly show STT states and transcribed text."""
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Position top-center
        screen = QApplication.primaryScreen().availableGeometry()
        self.setFixedSize(screen.width() - 200, 200)
        self.move(100, 50)
        
        layout = QVBoxLayout(self)
        self.container = QFrame()
        self.container.setStyleSheet("""
            background: rgba(0, 0, 0, 220);
            border: 2px solid #555555;
            border-radius: 20px;
        """)
        inner_layout = QVBoxLayout(self.container)
        
        self.status_lbl = QLabel("STT: IDLE")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_lbl.setStyleSheet("color: #00FFFF; font-size: 32px; font-weight: bold; border: none; background: transparent;")
        inner_layout.addWidget(self.status_lbl)
        
        self.text_lbl = QLabel("(Speak to transcribe)")
        self.text_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_lbl.setWordWrap(True)
        self.text_lbl.setStyleSheet("color: #FFFFFF; font-size: 28px; border: none; background: transparent;")
        inner_layout.addWidget(self.text_lbl)
        
        self.mic_lbl = QLabel("Mic Level: [░░░░░░░░░░░░░░░░░░░░] 0%")
        self.mic_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mic_lbl.setStyleSheet("color: #AAAAAA; f ont-size: 18px; border: none; background: transparent; font-family: monospace;")
        inner_layout.addWidget(self.mic_lbl)
        
        layout.addWidget(self.container)
        self.show()
        
    def update_state(self, state_text, color):
        self.status_lbl.setText(state_text)
        self.status_lbl.setStyleSheet(f"color: {color}; font-size: 32px; font-weight: bold; border: none; background: transparent;")
        
    def update_transcription(self, text):
        if text:
            self.text_lbl.setText(f'"{text}"')
        
    def update_mic(self, rms):
        # With AGC, RMS should be around 0.05 to 0.2 when talking
        pct = min(int(rms * 500), 100)
        bars = int(pct / 5)
        bar_str = "█" * bars + "░" * (20 - bars)
        self.mic_lbl.setText(f"Mic Level: [{bar_str}] {pct}%")

    def show_message(self, text, color, duration_ms=3000):
        self.update_state(text, color)


# ════════════════════════════════════════════════════════════════
#  CHAT BUBBLE
# ════════════════════════════════════════════════════════════════
class ChatBubble(QFrame):
    def __init__(self, text, is_user=True, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setFixedWidth(260)
        if is_user:
            lbl.setStyleSheet("""
                background-color: rgba(0,200,100,40);
                color: white;
                border: 1px solid rgba(0,200,100,80);
                border-radius: 12px;
                padding: 8px;
                font-size: 12px;
            """)
            layout.addStretch()
            layout.addWidget(lbl)
        else:
            lbl.setStyleSheet("""
                background-color: rgba(255,255,255,15);
                color: rgba(255,255,255,210);
                border: 1px solid rgba(255,255,255,30);
                border-radius: 12px;
                padding: 8px;
                font-size: 12px;
            """)
            layout.addWidget(lbl)
            layout.addStretch()


# ════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ════════════════════════════════════════════════════════════════
class ZeroTouchV0(QMainWindow):
    """
    Main window for Zero Touch v0.

    Layout (top → bottom):
      ┌──────────────────────────────┐  ← coloured border: RED=sleep / GREEN=wake
      │  title bar  (drag handle)    │
      │  camera feed  (320×240)      │
      │  status pill                 │
      │  chat log  (scrollable)      │
      │  STT status bar              │
      └──────────────────────────────┘
    """

    def __init__(self):
        super().__init__()
        self._drag_pos  = QPoint()
        self._wake      = False          # False = SLEEP, True = WAKE
        self._stt_state = "IDLE"
        self._pulse_val = 0.0

        self._stt_overlay = STTOverlayWindow()

        self._setup_ui()
        self._start_threads()

        # Pulse animation timer
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._tick_pulse)
        self._pulse_timer.start(50)

    # ── UI SETUP ──────────────────────────────────────────────
    def _setup_ui(self):
        self.setWindowTitle("Zero Touch v0")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(360, 640)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # Allow capturing Space bar

        # Position: bottom-right corner
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 380, screen.height() - 660)

        # ── Central container ────────────────────────────────
        self._container = QFrame(self)
        self._container.setGeometry(0, 0, 360, 640)
        self._container.setObjectName("container")

        root = QVBoxLayout(self._container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Title bar ────────────────────────────────────────
        title_bar = QFrame()
        title_bar.setFixedHeight(48)
        title_bar.setStyleSheet("background: transparent;")
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(16, 0, 12, 0)

        self._title_lbl = QLabel("Zero Touch v0")
        self._title_lbl.setStyleSheet(
            "color: white; font-size: 15px; font-weight: bold; background: transparent;"
        )
        tb_layout.addWidget(self._title_lbl)
        tb_layout.addStretch()

        self._state_badge = QLabel("SLEEP")
        self._state_badge.setStyleSheet(
            "color: #FF4444; font-size: 11px; font-weight: bold;"
            " background: rgba(255,68,68,20); border: 1px solid rgba(255,68,68,60);"
            " border-radius: 8px; padding: 2px 8px;"
        )
        tb_layout.addWidget(self._state_badge)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            "QPushButton{color:rgba(255,255,255,150);border:none;background:transparent;font-size:14px;}"
            "QPushButton:hover{color:#FF4444;}"
        )
        close_btn.clicked.connect(self.close)
        tb_layout.addWidget(close_btn)
        root.addWidget(title_bar)

        # ── Camera feed ──────────────────────────────────────
        self._cam_label = QLabel()
        self._cam_label.setFixedSize(360, 270)
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setStyleSheet(
            "background: rgba(0,0,0,120); border-radius: 0px; color: rgba(255,255,255,60); font-size: 13px;"
        )
        self._cam_label.setText("Camera initialising…")
        root.addWidget(self._cam_label)

        # ── Gesture status pill ──────────────────────────────
        self._gest_pill = QLabel("Gesture: —  |  FPS: —")
        self._gest_pill.setFixedHeight(32)
        self._gest_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gest_pill.setStyleSheet(
            "background: rgba(255,255,255,8); color: rgba(255,255,255,160);"
            " font-size: 11px; border-top: 1px solid rgba(255,255,255,20);"
            " border-bottom: 1px solid rgba(255,255,255,20);"
        )
        root.addWidget(self._gest_pill)

        # ── Chat log ─────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll.verticalScrollBar().setStyleSheet("QScrollBar{width:0px;}")

        self._chat_inner = QWidget()
        self._chat_inner.setStyleSheet("background: transparent;")
        self._chat_layout = QVBoxLayout(self._chat_inner)
        self._chat_layout.setContentsMargins(8, 8, 8, 8)
        self._chat_layout.setSpacing(6)
        self._chat_layout.addStretch()
        scroll.setWidget(self._chat_inner)
        self._scroll = scroll
        root.addWidget(scroll)

        # ── PTT Button ───────────────────────────────────────
        self._ptt_btn = QPushButton("Hold SPACE to Talk")
        self._ptt_btn.setFixedHeight(40)
        self._ptt_btn.setStyleSheet(
            "QPushButton { background: #333333; color: white; border-radius: 5px; font-weight: bold; margin: 5px 10px; }"
            "QPushButton:pressed { background: #555555; color: #00FFFF; }"
        )
        self._ptt_btn.pressed.connect(self._on_ptt_pressed)
        self._ptt_btn.released.connect(self._on_ptt_released)
        root.addWidget(self._ptt_btn)

        # ── STT status bar & Mic Volume ───────────────────────
        stt_layout = QVBoxLayout()
        stt_layout.setContentsMargins(0, 0, 0, 0)
        stt_layout.setSpacing(0)
        
        self._stt_bar = QLabel("STT: loading…")
        self._stt_bar.setFixedHeight(24)
        self._stt_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stt_bar.setStyleSheet(
            "background: rgba(0,0,0,80); color: rgba(255,255,255,120);"
            " font-size: 11px; border-top: 1px solid rgba(255,255,255,15);"
        )
        stt_layout.addWidget(self._stt_bar)

        # Mic volume visualizer
        self._mic_bar = QFrame()
        self._mic_bar.setFixedHeight(6)
        self._mic_bar.setStyleSheet("background: #222222; border-bottom-left-radius: 20px; border-bottom-right-radius: 20px;")
        
        self._mic_level = QFrame(self._mic_bar)
        self._mic_level.setGeometry(0, 0, 0, 6)
        self._mic_level.setStyleSheet("background: #00FFFF; border-bottom-left-radius: 20px;")
        
        stt_layout.addWidget(self._mic_bar)
        
        stt_container = QWidget()
        stt_container.setLayout(stt_layout)
        stt_container.setFixedHeight(30)
        root.addWidget(stt_container)

        self._add_chat("System", "Zero Touch v0 started. Say 'Hey Jarvis' or hold Open Palm to wake.", is_user=False)

    # ── THREADS ───────────────────────────────────────────────
    def _start_threads(self):
        # Gesture thread
        self._gest_thread = GestureThread()
        self._gest_thread.frame_ready.connect(self._on_frame)
        self._gest_thread.wake_requested.connect(self._on_gesture_wake)
        self._gest_thread.sleep_requested.connect(self._on_gesture_sleep)
        self._gest_thread.start()

        # STT thread
        self._stt_thread = STTThread()
        self._stt_thread.wake_detected.connect(self._on_stt_wake)
        self._stt_thread.transcription_ready.connect(self._on_transcription)
        self._stt_thread.status_changed.connect(self._on_stt_status)
        self._stt_thread.mic_volume.connect(self._update_mic_volume)
        self._stt_thread.start()

    # ── GESTURE CALLBACKS ─────────────────────────────────────
    def _on_frame(self, cv_img, state, progress, mlp_gest, fps):
        try:
            import cv2
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (360, 270))
            h, w, ch = rgb.shape
            qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self._cam_label.setPixmap(QPixmap.fromImage(qt_img))
        except Exception:
            pass

        self._gest_pill.setText(
            f"Gesture: {mlp_gest}  |  Clutch: {state}  |  FPS: {int(fps)}"
        )

    def _on_gesture_wake(self):
        if not self._wake:
            self._set_wake(True)
            self._add_chat("System", "Woken by gesture (Open Palm hold).", is_user=False)

    def _on_gesture_sleep(self):
        if self._wake:
            self._set_wake(False)
            self._add_chat("System", "Back to sleep (Fist hold).", is_user=False)

    # ── STT CALLBACKS ─────────────────────────────────────────
    def _on_stt_wake(self):
        """Wake word detected by openwakeword."""
        if not self._wake:
            self._set_wake(True)
            self._add_chat("System", "Woken by voice (Hey Jarvis).", is_user=False)

    def _on_stt_status(self, status: str):
        self._stt_state = status
        labels = {
            "IDLE":      "STT: Waiting for wake word…",
            "LISTENING": "STT: Listening — speak now…",
            "THINKING":  "STT: Transcribing…",
        }
        self._stt_bar.setText(labels.get(status, f"STT: {status}"))

        colors = {
            "IDLE":      "rgba(255,255,255,120)",
            "LISTENING": "#FFFF44",
            "THINKING":  "#FF8C00",
        }
        
        # Update floating overlay
        if status == "LISTENING":
            self._stt_overlay.update_state("🎙️ Listening... (Speak now)", "#FFFF44")
        elif status == "THINKING":
            self._stt_overlay.update_state("⏳ Transcribing...", "#FF8C00")
        elif status == "IDLE":
            if self._wake:
                self._stt_overlay.update_state("🟢 Awake (Listening for commands...)", "#44FF88")
            else:
                self._stt_overlay.update_state("STT: Waiting for 'Jarvis'...", "#00FFFF")

        c = colors.get(status, "white")
        self._stt_bar.setStyleSheet(
            f"background: rgba(0,0,0,80); color: {c};"
            " font-size: 11px; font-weight: bold;"
            " border-top: 1px solid rgba(255,255,255,15);"
        )

    def _update_mic_volume(self, rms: float):
        """Update the green mic level bar at the very bottom."""
        self._stt_overlay.update_mic(rms)
        
        # We scale it so 0.01 is around 50% width since we use raw RMS now
        width_ratio = min(rms * 50.0, 1.0)
        max_width = self._mic_bar.width()
        bar_width = int(max_width * width_ratio)
        
        color = "#00FFFF"
        self._mic_level.setGeometry(0, 0, bar_width, 6)
        self._mic_level.setStyleSheet(f"background: {color}; border-bottom-left-radius: 20px;")

    def _on_transcription(self, text: str):
        """Got transcribed speech. Check for sleep words, else send to OpenClaw."""
        self._stt_overlay.update_transcription(text)
        text_lower = text.lower()

        # Check for WAKE word dynamically via Whisper
        # Whisper might misspell "Jarvis", especially with the Indonesian language setting
        wake_variants = ["jarvis", "jervis", "darfis", "darvis", "harvis", "garvis", "servis"]
        if not self._wake and any(w in text_lower for w in wake_variants):
            self._add_chat("You (voice)", text, is_user=True)
            self._stt_overlay.show_message("🟢 System Awake", "#44FF88", 3000)
            self._set_wake(True)
            self._add_chat("System", "Woken by voice (detected 'jarvis').", is_user=False)
            self._speak("Halo, ada yang bisa saya bantu?")
            return

        # Check sleep command
        if any(w in text_lower for w in SLEEP_WORDS):
            self._add_chat("You (voice)", text, is_user=True)
            self._set_wake(False)
            self._add_chat("System", "Going to sleep…", is_user=False)
            self._speak("Baik, saya akan tidur sekarang")
            return

        # Only send to OpenClaw if in WAKE state
        if self._wake:
            self._add_chat("You (voice)", text, is_user=True)
            text_lower = text.lower()
            
            import re
            import difflib
            
            matched = next((app for kw, app in APP_KEYWORDS.items() if kw in text_lower), None)
            file_match = re.search(r'(?:buka|open|tampilkan)\s+([a-zA-Z0-9_\-\.\s]+)', text_lower)
            
            if matched:
                ok = send_to_openclaw(text)
                if ok:
                    status = f"✅ Launching {matched}..."
                    self._speak(f"Membuka {matched}")
                else:
                    status = "❌ Failed to reach host_bridge.py — is it running?"
                self._add_chat("System", status, is_user=False)
            elif file_match:
                workspace_dir = r"E:\OpenClawProject\test_folder"
                target_word = file_match.group(1).strip()
                try:
                    files = [f for f in os.listdir(workspace_dir) if os.path.isfile(os.path.join(workspace_dir, f))]
                    # Low cutoff for aggressive matching (images.gpg -> images.jpg)
                    matches = difflib.get_close_matches(target_word, files, n=1, cutoff=0.3)
                    
                    if matches:
                        best_match = matches[0]
                        try:
                            r = requests.post(HOST_BRIDGE_URL, json={"action": "open", "file": best_match}, timeout=5)
                            if r.ok:
                                status = f"✅ Opening file {best_match}..."
                                self._speak(f"Membuka file {best_match}")
                            else:
                                status = "❌ Failed to reach host_bridge.py"
                        except Exception:
                            status = "❌ Failed to reach host_bridge.py"
                        self._add_chat("System", status, is_user=False)
                    else:
                        self._generate_and_speak(text)
                except Exception as e:
                    self._generate_and_speak(text)
            else:
                self._generate_and_speak(text)
        else:
            # In sleep state, STT heard something, but we just ignore it (already shown on overlay)
            self._add_chat("System", f"[SLEEP] Heard: {text!r} – ignored.", is_user=False)

    def _generate_and_speak(self, text: str):
        # Run in background thread to avoid freezing UI
        threading.Thread(target=self._llm_tts_worker, args=(text,), daemon=True).start()

    def _speak(self, text: str):
        # Dedicated TTS-only path
        threading.Thread(target=self._tts_worker, args=(text,), daemon=True).start()

    def _tts_worker(self, text: str):
        try:
            import uuid
            import os
            out_file = f"tts_out_{uuid.uuid4().hex[:6]}.wav"
            
            from tts import synthesize_indonesian
            synthesize_indonesian(text, out_file)
            
            # Play audio using winsound
            import winsound
            if os.path.exists(out_file):
                winsound.PlaySound(out_file, winsound.SND_FILENAME)
                os.remove(out_file)
        except Exception as e:
            self._add_chat("System", f"❌ TTS Error: {e}", is_user=False)

    def _llm_tts_worker(self, prompt: str):
        self._add_chat("System", "Thinking...", is_user=False)
        try:
            r = requests.post("http://localhost:11434/api/generate", json={
                "model": "llama3.2:latest",
                "prompt": prompt,
                "stream": False,
                "system": "Anda adalah Jarvis, asisten AI untuk dokter bedah ProTel. Jawab singkat dan padat dalam bahasa Indonesia."
            }, timeout=30)
            
            if r.ok:
                response_text = r.json().get("response", "").strip()
                self._add_chat("Jarvis", response_text, is_user=False)
                # Play it!
                self._tts_worker(response_text)
            else:
                self._add_chat("System", f"❌ LLM error: {r.status_code}", is_user=False)
                
        except Exception as e:
            self._add_chat("System", f"❌ LLM/TTS Error: {e}", is_user=False)

    # ── STATE MANAGEMENT ──────────────────────────────────────
    def _set_wake(self, wake: bool):
        self._wake = wake
        self._gest_thread.set_active(wake)
        self._stt_thread.set_force_listen(wake)  # Tell STT to bypass wake word!
        self.update()           # trigger paintEvent for border repaint

        if wake:
            self._state_badge.setText("WAKE")
            self._state_badge.setStyleSheet(
                "color: #44FF88; font-size: 11px; font-weight: bold;"
                " background: rgba(68,255,136,20); border: 1px solid rgba(68,255,136,60);"
                " border-radius: 8px; padding: 2px 8px;"
            )
        else:
            self._state_badge.setText("SLEEP")
            self._state_badge.setStyleSheet(
                "color: #FF4444; font-size: 11px; font-weight: bold;"
                " background: rgba(255,68,68,20); border: 1px solid rgba(255,68,68,60);"
                " border-radius: 8px; padding: 2px 8px;"
            )

    # ── PUSH TO TALK HANDLERS ─────────────────────────────────
    def _on_ptt_pressed(self):
        self._ptt_btn.setStyleSheet("QPushButton { background: #555555; color: #00FFFF; border-radius: 5px; font-weight: bold; margin: 5px 10px; }")
        self._ptt_btn.setText("🎙️ Listening...")
        if hasattr(self, "_stt_thread"):
            self._stt_thread.ptt_start()

    def _on_ptt_released(self):
        self._ptt_btn.setStyleSheet("QPushButton { background: #333333; color: white; border-radius: 5px; font-weight: bold; margin: 5px 10px; }")
        self._ptt_btn.setText("Hold SPACE to Talk")
        if hasattr(self, "_stt_thread"):
            self._stt_thread.ptt_stop()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._on_ptt_pressed()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._on_ptt_released()
        super().keyReleaseEvent(event)

    # ── CHAT LOG HELPER ───────────────────────────────────────
    def _add_chat(self, sender: str, text: str, is_user: bool):
        display = f"[{sender}] {text}"
        bubble  = ChatBubble(display, is_user=is_user)
        self._chat_layout.insertWidget(self._chat_layout.count() - 1, bubble)
        QTimer.singleShot(80, self._scroll_bottom)

    def _scroll_bottom(self):
        try:
            self._scroll.verticalScrollBar().setValue(
                self._scroll.verticalScrollBar().maximum()
            )
        except RuntimeError:
            pass

    # ── ANIMATION ─────────────────────────────────────────────
    def _tick_pulse(self):
        self._pulse_val = (math.sin(time.time() * 4) + 1) / 2
        self.update()

    # ── PAINTING (border + background) ────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background gradient
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor(18, 18, 22, 245))
        grad.setColorAt(1, QColor(10, 10, 14, 245))
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

        # Coloured border (pulsing)
        if self._wake:
            alpha  = int(160 + 95 * self._pulse_val)
            border = QColor(68, 255, 136, alpha)
            width  = 2.5 + 1.5 * self._pulse_val
        else:
            alpha  = int(140 + 60 * self._pulse_val)
            border = QColor(255, 68, 68, alpha)
            width  = 2.0

        pen = QPen(border, width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            self.rect().adjusted(1, 1, -1, -1), 20, 20
        )

    # ── DRAG ──────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    # ── CLEANUP ───────────────────────────────────────────────
    def closeEvent(self, event):
        print("[app] Shutting down…")
        self._stt_overlay.close()
        self._pulse_timer.stop()
        self._gest_thread.stop()
        self._stt_thread.stop()
        event.accept()


# ════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply Inter / Segoe UI font
    font = QFont("Inter")
    if not font.exactMatch():
        font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = ZeroTouchV0()
    window.show()
    sys.exit(app.exec())

# Tambahin di souls kalau ketemu file .txt atau .docs atau tipe2 lainnya, langsung tahu harus pakai tools apa