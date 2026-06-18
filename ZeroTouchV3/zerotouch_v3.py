"""
zerotouch_v1.py  –  Zero Touch v1  (OpenClaw Edition)
=======================================================
Upgrades v0's hardcoded keyword pipeline to a full agentic pipeline:

  Voice → Whisper STT → Llama 3.1 (OpenClaw) → tool / answer
                                              ↓
                                    Llama 3.2 (voice formatter)
                                              ↓
                                          TTS playback

New pipeline (vs v0):
  • No APP_KEYWORDS dict — Llama 3.1 understands intent naturally.
  • Supports advanced commands: "zoom in top-right 2x", "open excel", etc.
  • Llama 3.2 role changed: voice-formatter only (short, natural Indonesian).
  • Tool execution + TTS generation run in parallel (concurrent.futures).
  • Falls back to direct Llama 3.2 generation if OpenClaw is unreachable.

Usage:
    cd e:\\OpenclawMainFolder
    python ZeroTouchV1\\zerotouch_v1.py

Pre-flight check (run first):
    python ZeroTouchV1\\test_openclaw.py
"""

import sys
import os
import time
import threading
import queue
import math
import logging
import warnings
import subprocess
import concurrent.futures

os.environ["PYTHONIOENCODING"]          = "utf-8"
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import requests

# ── Path resolution ───────────────────────────────────────────────
# zerotouch_v1.py lives in ZeroTouchV1/, so ROOT is one level up
V3_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(V3_DIR)

ZEROTOUCH_DIR = os.path.join(ROOT, "ZeroTouch")
STT_DIR       = os.path.join(ROOT, "VoiceSetting", "zerotouch_voicerecognition")

sys.path.insert(0, V3_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, ZEROTOUCH_DIR)
sys.path.insert(0, STT_DIR)

import emr_rag

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QFrame, QScrollArea, QPushButton,
)
from PyQt6.QtCore  import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui   import (
    QColor, QPainter, QPen, QFont, QLinearGradient, QImage, QPixmap
)

# ── Config ────────────────────────────────────────────────────────
HOST_BRIDGE_V1_URL  = "http://127.0.0.1:5003/run"
SCREEN_CONTROL_URL  = "http://127.0.0.1:5003/screen"

STT_MODEL_SIZE   = "small"
STT_DEVICE       = "cpu"
STT_COMPUTE      = "int8"
STT_THREADS      = 16
STT_LANGUAGE     = "id"
WAKE_WORD_MODEL  = os.path.join(STT_DIR, "models", "hello_zerotouch_v2.onnx")
WAKE_WORD_THRESH = 0.85
SAMPLE_RATE      = 16000
OWW_CHUNK        = 1280
SILENCE_THRESHOLD    = 0.03
# How long the STT waits for silence before it stops recording and transcribes.
# Keep this identical to stt_activation.py (3.5s) — this is the RECORDING cutoff.
SILENCE_TIMEOUT      = 3.5
# How long (seconds) the assistant stays in voice-wake state without any NEW
# speech activity.  Controlled by a QTimer in ZeroTouchV1, NOT by SILENCE_TIMEOUT.
VOICE_WAKE_TIMEOUT   = 15.0
# Minimum peak RMS across the whole recording for it to be considered real speech.
# Below this value the audio is likely background noise → skip Whisper entirely.
# Set conservatively low (0.015) to avoid rejecting soft-spoken microphone input.
MIN_SPEECH_RMS       = 0.015
# Whisper assigns a no_speech_prob [0-1] to each segment.  Segments above this
# value are hallucinations / filler and are discarded before joining the text.
# Keep this HIGH (≥ 0.85) — real speech often scores 0.7-0.82, so a low threshold
# will incorrectly drop legitimate transcriptions like "Halo ZeroTouch".
WHISPER_NO_SPEECH_THRESH = 0.85
# Phrases Whisper commonly hallucinates on near-silent Indonesian audio.
# Comparison is done lower-case, stripped, after joining segments.
_HALLUCINATION_BLOCKLIST = {
    "terima kasih",
    "terima kasih karena monoton",
    "terima kasih telah menonton",
    "terima kasih sudah menonton",
    "terima kasih.",
    "subtitle oleh komunitas amara",
    "subtitle by komunitas amara",
    "semoga bermanfaat",
    ". . .",
    "...",
    "[musik]",
    "[music]",
}

SLEEP_GESTURE_HOLD = 2.0
SLEEP_WORDS = ["sleep", "tidur", "stop listening", "berhenti", "diam"]

APP_KEYWORDS = {
    "microsoft word": "word", "word": "word", "winword": "word", "buka word": "word",
    "whatsapp": "whatsapp", "buka whatsapp": "whatsapp", "watsap": "whatsapp",
    "notepad": "notepad", "buka notepad": "notepad", "buka catatan": "notepad",
    "excel": "excel", "buka excel": "excel",
    "powerpoint": "powerpoint", "buka powerpoint": "powerpoint",
    "chrome": "chrome", "buka chrome": "chrome", "buka browser": "chrome"
}


# ════════════════════════════════════════════════════════════════
#  GESTURE THREAD  (identical to v0)
# ════════════════════════════════════════════════════════════════
class GestureThread(QThread):
    frame_ready     = pyqtSignal(np.ndarray, str, float, str, float)
    wake_requested  = pyqtSignal()
    sleep_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._running = False
        self._fist_hold_start = None

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
        if state == "ACTIVE":
            self.wake_requested.emit()
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
                self._engine.clutch.state = "ACTIVE"
                self._engine.clutch.grace_period = 999999.0
            else:
                self._engine.clutch.state = "IDLE"
                self._engine.clutch.progress = 0
                self._engine.clutch.grace_period = 1.0

    def stop(self):
        self._running = False
        if hasattr(self, "_engine"):
            self._engine.running = False
        self.wait(3000)


# ════════════════════════════════════════════════════════════════
#  STT THREAD  (identical to v0)
# ════════════════════════════════════════════════════════════════
class STTThread(QThread):
    wake_detected       = pyqtSignal()
    transcription_ready = pyqtSignal(str)
    status_changed      = pyqtSignal(str)
    mic_volume          = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self._running      = True
        self._force_listen = False
        self._voice_awake  = False
        self._ptt_active   = threading.Event()
        self._ptt_stop     = threading.Event()

    def set_force_listen(self, force: bool):
        self._force_listen = force

    def set_voice_awake(self, awake: bool):
        self._voice_awake = awake

    def ptt_start(self):
        self._ptt_active.set()
        self._ptt_stop.clear()

    def ptt_stop(self):
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

        state           = S.IDLE
        buf             = []
        last_speech     = 0.0
        bg_rms          = 0.001
        session_max_rms = 0.0

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=OWW_CHUNK,
                                callback=_cb)
        self.status_changed.emit("IDLE")

        trigger_count = 0
        REQUIRED_TRIGGERS = 5

        with stream:
            while self._running:
                try:
                    block = audio_q.get(timeout=0.1)
                except queue.Empty:
                    # Check for PTT press even when no audio block is ready
                    if state == S.IDLE and self._ptt_active.is_set():
                        state = S.LISTENING
                        buf   = []
                        last_speech = time.time()
                        session_max_rms = 0.0
                        self.status_changed.emit("LISTENING")
                    continue

                mono     = block[:, 0]
                rms_raw  = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
                self.mic_volume.emit(rms_raw)

                # ── IDLE: wake word detection ──
                if state == S.IDLE:
                    # Exact same conversion as stt_activation._float32_to_int16
                    chunk_i16   = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
                    predictions = oww.predict(chunk_i16)
                    wake_triggered = False
                    for ww_name, score in predictions.items():
                        if score >= WAKE_WORD_THRESH:
                            trigger_count += 1
                            if trigger_count >= REQUIRED_TRIGGERS:
                                wake_triggered = True
                                self.wake_detected.emit()
                                trigger_count = 0
                                break
                        else:
                            trigger_count = max(0, trigger_count - 1)
                    if self._ptt_active.is_set() or self._force_listen or wake_triggered or self._voice_awake:
                        state           = S.LISTENING
                        buf             = [mono.copy()]  # raw audio, same as stt_activation
                        last_speech     = time.time()
                        session_max_rms = rms_raw
                        self.status_changed.emit("LISTENING")

                # ── LISTENING: record until silence or PTT release ──
                elif state == S.LISTENING:
                    buf.append(mono.copy())  # raw audio, same as stt_activation
                    if rms_raw > session_max_rms:
                        session_max_rms = rms_raw
                    if rms_raw >= SILENCE_THRESHOLD:
                        last_speech = time.time()

                    ptt_released           = self._ptt_stop.is_set() and not self._ptt_active.is_set()
                    silence_dur            = time.time() - last_speech
                    silence_timeout_reached = silence_dur >= SILENCE_TIMEOUT and not self._ptt_active.is_set()

                    if ptt_released or silence_timeout_reached:
                        self.status_changed.emit("THINKING")
                        audio_data = np.concatenate(buf) if buf else np.array([], dtype=np.float32)

                        # ── Layer 1: Pre-flight RMS gate ──────────────────────────────────
                        # If the entire recording was too quiet, Whisper will hallucinate
                        # filler phrases (e.g. "Terima kasih") instead of returning empty.
                        # Skip the transcription call entirely for near-silent audio.
                        if session_max_rms < MIN_SPEECH_RMS:
                            print(f"[stt] Skipped (peak RMS {session_max_rms:.4f} < {MIN_SPEECH_RMS}) — no real speech detected.")
                            text = ""
                        else:
                            # ── Layer 2: Whisper with no_speech_threshold ─────────────────
                            # no_speech_threshold: segments whose no_speech_prob exceeds
                            # this value are suppressed by Whisper before we even see them.
                            # Params intentionally mirrored from the working stt_activation.py:
                            # • condition_on_previous_text=False  → prevents context bleed
                            #   between sessions which causes hallucinations.
                            # • vad_parameters threshold=0.5      → tighter VAD gate on
                            #   what counts as speech vs silence.
                            segs, _ = stt.transcribe(
                                audio_data, beam_size=5, language=STT_LANGUAGE,
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=400, threshold=0.5),
                                condition_on_previous_text=False,
                                no_speech_threshold=WHISPER_NO_SPEECH_THRESH,
                                # initial_prompt="percakapan, bahasa indonesia, kata formal, kata informal, kalimat valid",
                            )
                            # ── Layer 2b: Per-segment no_speech_prob filter ───────────────
                            # Whisper exposes no_speech_prob on each segment.  Drop any
                            # segment that looks like silence / hallucination.
                            good_parts = []
                            for s in segs:
                                nsp = getattr(s, "no_speech_prob", 0.0)
                                if nsp < WHISPER_NO_SPEECH_THRESH:
                                    good_parts.append(s.text.strip())
                                else:
                                    print(f"[stt] Dropped segment (no_speech_prob={nsp:.2f}): {s.text.strip()!r}")
                            text = " ".join(good_parts).strip()

                            # ── Layer 3: Post-transcription hallucination blocklist ────────
                            # Known Whisper filler phrases on Indonesian silent audio.
                            if text.lower().strip(" .") in {p.strip(" .") for p in _HALLUCINATION_BLOCKLIST}:
                                print(f"[stt] Blocked hallucination: {text!r}")
                                text = ""

                        print(f"[stt] Transcribed: {text!r}  (peak_rms={session_max_rms:.4f})")
                        self.transcription_ready.emit(text if text else "(nothing heard)")
                        buf             = []
                        state           = S.IDLE
                        self._force_listen = False
                        self._ptt_stop.clear()
                        session_max_rms = 0.0
                        if hasattr(oww, "reset"):  # FIX 1 from stt_activation
                            oww.reset()
                        with audio_q.mutex:         # FIX 2 from stt_activation
                            audio_q.queue.clear()
                        self.status_changed.emit("IDLE")

    def stop(self):
        self._running = False
        self.wait(5000)


# ════════════════════════════════════════════════════════════════
#  STT FLOATING OVERLAY  (identical to v0)
# ════════════════════════════════════════════════════════════════
class STTOverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        screen = QApplication.primaryScreen().availableGeometry()
        self.setFixedSize(screen.width() - 200, 200)
        self.move(100, 50)

        layout    = QVBoxLayout(self)
        self.container = QFrame()
        self.container.setStyleSheet(
            "background: rgba(0,0,0,75); border: 2px solid rgba(255,255,255,40); border-radius: 20px;"
        )
        inner = QVBoxLayout(self.container)

        self.status_lbl = QLabel("STT: IDLE")
        self.status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_lbl.setStyleSheet(
            "color: #00FFFF; font-size: 32px; font-weight: bold; border: none; background: transparent;"
        )
        inner.addWidget(self.status_lbl)

        self.text_lbl = QLabel("(Speak to transcribe)")
        self.text_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_lbl.setWordWrap(True)
        self.text_lbl.setStyleSheet(
            "color: #FFFFFF; font-size: 28px; border: none; background: transparent;"
        )
        inner.addWidget(self.text_lbl)

        self.mic_lbl = QLabel("Mic Level: [░░░░░░░░░░░░░░░░░░░░] 0%")
        self.mic_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mic_lbl.setStyleSheet(
            "color: #AAAAAA; font-size: 18px; border: none; background: transparent; font-family: monospace;"
        )
        inner.addWidget(self.mic_lbl)
        layout.addWidget(self.container)
        self.show()

    def update_state(self, text, color):
        self.status_lbl.setText(text)
        self.status_lbl.setStyleSheet(
            f"color: {color}; font-size: 32px; font-weight: bold; border: none; background: transparent;"
        )

    def update_transcription(self, text):
        if text:
            self.text_lbl.setText(f'"{text}"')

    def update_mic(self, rms):
        pct     = min(int(rms * 500), 100)
        bars    = int(pct / 5)
        bar_str = "█" * bars + "░" * (20 - bars)
        self.mic_lbl.setText(f"Mic Level: [{bar_str}] {pct}%")

    def show_message(self, text, color, duration_ms=3000):
        self.update_state(text, color)


# ════════════════════════════════════════════════════════════════
#  CHAT BUBBLE  (identical to v0)
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
            lbl.setStyleSheet(
                "background-color: rgba(0,200,100,40); color: white;"
                " border: 1px solid rgba(0,200,100,80); border-radius: 12px;"
                " padding: 8px; font-size: 12px;"
            )
            layout.addStretch()
            layout.addWidget(lbl)
        else:
            lbl.setStyleSheet(
                "background-color: rgba(255,255,255,15); color: rgba(255,255,255,210);"
                " border: 1px solid rgba(255,255,255,30); border-radius: 12px;"
                " padding: 8px; font-size: 12px;"
            )
            layout.addWidget(lbl)
            layout.addStretch()


# ════════════════════════════════════════════════════════════════
#  MAIN WINDOW  –  ZeroTouch V1
# ════════════════════════════════════════════════════════════════
class ZeroTouchV3(QMainWindow):
    chat_signal = pyqtSignal(str, str, bool)

    def __init__(self):
        super().__init__()
        self.chat_signal.connect(self._do_add_chat)
        self._drag_pos   = QPoint()
        self._voice_wake = False
        self._gest_wake  = False
        self._stt_state = "IDLE"
        self._pulse_val = 0.0
        # Tracks the last time real speech was processed (used for wake-state timeout).
        # Reset on every successful non-empty transcription and on wake-up.
        self._last_voice_activity = 0.0
        # Short-term memory for the LLM during a single WAKE session.
        # Cleared when voice goes to SLEEP.
        self._session_history = []

        # Auto-start host_bridge_v3.py (port 5003)
        try:
            self._bridge_proc = subprocess.Popen(
                [sys.executable, os.path.join(V3_DIR, "host_bridge_v3.py")],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
            print("[v3] Started host_bridge_v3.py on port 5003")
        except Exception as e:
            print(f"[v3] Failed to start host_bridge_v3.py: {e}")
            self._bridge_proc = None

        self._stt_overlay = STTOverlayWindow()
        emr_rag.initialize()
        self._setup_agent()

        self._setup_ui()
        self._start_threads()

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._tick_pulse)
        self._pulse_timer.start(50)

        # Voice-wake inactivity timer — fires every second and auto-sleeps voice
        # after VOICE_WAKE_TIMEOUT seconds of silence.  This is SEPARATE from
        # SILENCE_TIMEOUT (which only controls how long the STT records).
        self._wake_timer = QTimer(self)
        self._wake_timer.timeout.connect(self._check_voice_wake_timeout)
        self._wake_timer.start(1000)

    # ── UI SETUP ──────────────────────────────────────────────
    def _setup_ui(self):
        self.setWindowTitle("Zero Touch v3")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(360, 640)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.width() - 380, screen.height() - 660)

        self._container = QFrame(self)
        self._container.setGeometry(0, 0, 360, 640)
        self._container.setObjectName("container")
        root = QVBoxLayout(self._container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Title bar
        title_bar = QFrame()
        title_bar.setFixedHeight(48)
        title_bar.setStyleSheet("background: transparent;")
        tb = QHBoxLayout(title_bar)
        tb.setContentsMargins(16, 0, 12, 0)

        self._title_lbl = QLabel("Zero Touch v1  ⚡")
        self._title_lbl.setStyleSheet(
            "color: white; font-size: 15px; font-weight: bold; background: transparent;"
        )
        tb.addWidget(self._title_lbl)
        tb.addStretch()

        self._v_badge = QLabel("V: SLEEP")
        self._v_badge.setStyleSheet(
            "color: #FF4444; font-size: 11px; font-weight: bold;"
            " background: rgba(255,68,68,20); border: 1px solid rgba(255,68,68,60);"
            " border-radius: 8px; padding: 2px 8px;"
        )
        tb.addWidget(self._v_badge)

        self._g_badge = QLabel("G: SLEEP")
        self._g_badge.setStyleSheet(
            "color: #FF4444; font-size: 11px; font-weight: bold;"
            " background: rgba(255,68,68,20); border: 1px solid rgba(255,68,68,60);"
            " border-radius: 8px; padding: 2px 8px;"
        )
        tb.addWidget(self._g_badge)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            "QPushButton{color:rgba(255,255,255,150);border:none;background:transparent;font-size:14px;}"
            "QPushButton:hover{color:#FF4444;}"
        )
        close_btn.clicked.connect(self.close)
        tb.addWidget(close_btn)
        root.addWidget(title_bar)

        # Camera
        self._cam_label = QLabel()
        self._cam_label.setFixedSize(360, 270)
        self._cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label.setStyleSheet(
            "background: rgba(0,0,0,120); color: rgba(255,255,255,60); font-size: 13px;"
        )
        self._cam_label.setText("Camera initialising…")
        root.addWidget(self._cam_label)

        # Gesture pill
        self._gest_pill = QLabel("Gesture: —  |  FPS: —")
        self._gest_pill.setFixedHeight(32)
        self._gest_pill.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gest_pill.setStyleSheet(
            "background: rgba(255,255,255,8); color: rgba(255,255,255,160);"
            " font-size: 11px; border-top: 1px solid rgba(255,255,255,20);"
            " border-bottom: 1px solid rgba(255,255,255,20);"
        )
        root.addWidget(self._gest_pill)

        # Chat log
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        scroll.verticalScrollBar().setStyleSheet("QScrollBar{width:0px;}")
        self._chat_inner  = QWidget()
        self._chat_inner.setStyleSheet("background: transparent;")
        self._chat_layout = QVBoxLayout(self._chat_inner)
        self._chat_layout.setContentsMargins(8, 8, 8, 8)
        self._chat_layout.setSpacing(6)
        self._chat_layout.addStretch()
        scroll.setWidget(self._chat_inner)
        self._scroll = scroll
        root.addWidget(scroll)

        # PTT button
        self._ptt_btn = QPushButton("Hold SPACE to Talk")
        self._ptt_btn.setFixedHeight(40)
        self._ptt_btn.setStyleSheet(
            "QPushButton { background: #333333; color: white; border-radius: 5px;"
            " font-weight: bold; margin: 5px 10px; }"
            "QPushButton:pressed { background: #555555; color: #00FFFF; }"
        )
        self._ptt_btn.pressed.connect(self._on_ptt_pressed)
        self._ptt_btn.released.connect(self._on_ptt_released)
        root.addWidget(self._ptt_btn)

        # STT status bar + mic volume
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

        self._mic_bar   = QFrame()
        self._mic_bar.setFixedHeight(6)
        self._mic_bar.setStyleSheet(
            "background: #222222; border-bottom-left-radius: 20px; border-bottom-right-radius: 20px;"
        )
        self._mic_level = QFrame(self._mic_bar)
        self._mic_level.setGeometry(0, 0, 0, 6)
        self._mic_level.setStyleSheet("background: #00FFFF; border-bottom-left-radius: 20px;")
        stt_layout.addWidget(self._mic_bar)

        stt_container = QWidget()
        stt_container.setLayout(stt_layout)
        stt_container.setFixedHeight(30)
        root.addWidget(stt_container)

        self._add_chat("System", "Zero Touch v3 started. Local Llama 3.2 pipeline active.", is_user=False)

    # ── THREADS ───────────────────────────────────────────────
    def _start_threads(self):
        self._gest_thread = GestureThread()
        self._gest_thread.frame_ready.connect(self._on_frame)
        self._gest_thread.wake_requested.connect(self._on_gesture_wake)
        self._gest_thread.sleep_requested.connect(self._on_gesture_sleep)
        self._gest_thread.start()

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
        self._gest_pill.setText(f"Gesture: {mlp_gest}  |  Clutch: {state}  |  FPS: {int(fps)}")

    def _on_gesture_wake(self):
        if not self._gest_wake:
            self._set_gest_wake(True)
            self._add_chat("System", "Gesture Woken (Open Palm hold).", is_user=False)

    def _on_gesture_sleep(self):
        if self._gest_wake:
            self._set_gest_wake(False)
            self._add_chat("System", "Gesture Asleep (Fist hold).", is_user=False)

    # ── STT CALLBACKS ─────────────────────────────────────────
    def _on_stt_wake(self):
        if not self._voice_wake:
            self._set_voice_wake(True)
            self._add_chat("System", "Voice Woken (Hello Zero Touch).", is_user=False)
        # Always reset the inactivity clock on wake so the timeout counts from NOW.
        self._last_voice_activity = time.time()

    def _on_stt_status(self, status: str):
        self._stt_state = status
        labels = {
            "IDLE":      "STT: Waiting for wake word…",
            "LISTENING": "STT: Listening — speak now…",
            "THINKING":  "STT: Transcribing…",
        }
        self._stt_bar.setText(labels.get(status, f"STT: {status}"))
        colors = {"IDLE": "rgba(255,255,255,120)", "LISTENING": "#FFFF44", "THINKING": "#FF8C00"}
        if status == "LISTENING":
            self._stt_overlay.update_state("🎙️ Listening... (Speak now)", "#FFFF44")
        elif status == "THINKING":
            self._stt_overlay.update_state("⏳ Transcribing...", "#FF8C00")
        elif status == "IDLE":
            if self._voice_wake:
                self._stt_overlay.update_state("🟢 Voice Awake (Listening...)", "#44FF88")
            else:
                self._stt_overlay.update_state("STT: Waiting for 'Hello Zero Touch'...", "#00FFFF")
        c = colors.get(status, "white")
        self._stt_bar.setStyleSheet(
            f"background: rgba(0,0,0,80); color: {c};"
            " font-size: 11px; font-weight: bold;"
            " border-top: 1px solid rgba(255,255,255,15);"
        )

    def _update_mic_volume(self, rms: float):
        self._stt_overlay.update_mic(rms)
        width_ratio = min(rms * 50.0, 1.0)
        bar_width   = int(self._mic_bar.width() * width_ratio)
        self._mic_level.setGeometry(0, 0, bar_width, 6)
        self._mic_level.setStyleSheet("background: #00FFFF; border-bottom-left-radius: 20px;")

    # ── Voice-wake inactivity watchdog ─────────────────────────
    def _check_voice_wake_timeout(self):
        """Called every second by _wake_timer.  Puts voice to sleep automatically
        after VOICE_WAKE_TIMEOUT seconds of no real speech activity."""
        if not self._voice_wake:
            return  # already asleep, nothing to do
        if self._last_voice_activity == 0.0:
            return  # clock not started yet
        elapsed = time.time() - self._last_voice_activity
        # Show countdown in the STT bar when getting close to timeout
        remaining = VOICE_WAKE_TIMEOUT - elapsed
        if remaining <= 5.0:
            self._stt_bar.setText(f"STT: Voice sleep in {int(remaining)}s...")
        if elapsed >= VOICE_WAKE_TIMEOUT:
            print(f"[v1] Voice wake timeout after {elapsed:.0f}s of inactivity.")
            self._set_voice_wake(False)
            self._add_chat("System", f"Voice Asleep (no activity for {int(VOICE_WAKE_TIMEOUT)}s).", is_user=False)

    # ── TRANSCRIPTION → OPENCLAW PIPELINE ────────────────────
    def _on_transcription(self, text: str):
        is_ptt = getattr(self, "_last_was_ptt", False)
        self._last_was_ptt = False

        # ── Guard: ignore empty / noise-only results ──────────
        # Whisper sometimes returns an empty string or "(nothing heard)" when
        # there was silence or mic noise. Sending these to the LLM causes it
        # to hallucinate random tool calls. Drop them silently.
        # IMPORTANT: do NOT put voice to sleep here.  Voice sleep is handled
        # exclusively by _check_voice_wake_timeout (the VOICE_WAKE_TIMEOUT timer).
        # Sleeping on every empty result would kill wake state after the first
        # 3.5-second silence gap, making it impossible to hold a conversation.
        clean = text.strip()
        if not clean or clean == "(nothing heard)":
            print(f"[v1] Empty transcription ignored: {text!r}")
            return

        self._stt_overlay.update_transcription(clean)
        text_lower = clean.lower()

        # Wake-word variants (Whisper mishearings of "Hello Zero Touch")
        wake_variants = ["hello zero touch", "halo zero touch", "helo zero touch", "zero touch", "sero touch", "hello zero", "halo zero"]
        if not self._voice_wake and any(w in text_lower for w in wake_variants):
            self._add_chat("You (voice)", clean, is_user=True)
            self._set_voice_wake(True)
            self._last_voice_activity = time.time()  # start inactivity clock
            self._add_chat("System", "Voice Woken (detected 'hello zero touch').", is_user=False)
            self._speak("Halo, ada yang bisa saya bantu?")
            return

        # Sleep command
        if self._voice_wake and any(w in text_lower for w in SLEEP_WORDS):
            self._add_chat("You (voice)", clean, is_user=True)
            self._set_voice_wake(False)
            self._add_chat("System", "Voice Asleep...", is_user=False)
            self._speak("Baik, saya akan berhenti mendengarkan")
            return

        # WAKE state — route to processing pipeline
        if self._voice_wake or is_ptt:
            self._last_voice_activity = time.time()  # reset inactivity clock on every real command
            self._add_chat("You (voice)", clean, is_user=True)
            threading.Thread(
                target=self._process_pipeline, args=(clean,), daemon=True
            ).start()
        else:
            self._add_chat("System", f"[SLEEP] Heard: {clean!r} - ignored.", is_user=False)

    # ════════════════════════════════════════════════════════
    #  LANGCHAIN AGENT SETUP
    # ════════════════════════════════════════════════════════
    def _setup_agent(self):
        try:
            from langchain_agent import ZeroTouchAgent
            self.lc_agent = ZeroTouchAgent(
                execute_launcher_cb=self._execute_launcher,
                execute_screen_cb=self._execute_screen_control,
                find_patient_file_cb=self._find_patient_file,
                rag_query_cb=emr_rag.query,
                rag_ready_cb=emr_rag.is_ready
            )
            print("[v3] LangChain Agent initialized.")
        except Exception as e:
            print(f"[v3] Error initializing LangChain Agent: {e}")
            self.lc_agent = None

    # ════════════════════════════════════════════════════════
    #  LANGCHAIN PIPELINE EXECUTION
    # ════════════════════════════════════════════════════════
    def _process_pipeline(self, prompt: str):
        self._add_chat("System", "Thinking (LangChain)...", is_user=False)
        try:
            if not getattr(self, "lc_agent", None):
                self._add_chat("System", "LangChain Agent not initialized.", is_user=False)
                return
                
            output = self.lc_agent.process_prompt(prompt)
            self._add_chat("Jarvis", output, is_user=False)
            self._speak(output)
            
        except Exception as e:
            print(f"[LangChain] Error: {e}")
            self._add_chat("System", f"Agent Error: {e}", is_user=False)

    # ── Patient name extractor ────────────────────────────
    def _extract_patient_name(self, prompt: str) -> str:
        """Extract a patient name from a voice prompt using regex."""
        import re
        patterns = [
            r"(?:pasien|patient)\s+(?:bernama\s+)?([A-Za-z][A-Za-z\s]{2,30})(?:\s+(?:dengan|yang|untuk|nya)|[,.]|$)",
            r"(?:bernama|milik|untuk|buat)\s+([A-Za-z][A-Za-z\s]{2,30})(?:\s+(?:dengan|yang)|[,.]|$)",
        ]
        for pat in patterns:
            try:
                m = re.search(pat, prompt, re.IGNORECASE)
                if m:
                    name = m.group(1).strip()
                    if len(name) >= 3:
                        return name
            except re.error:
                pass
        # Fallback: look for known patient folders by name
        patient_root = os.path.join(V3_DIR, "data")
        if os.path.isdir(patient_root):
            p_lower = prompt.lower()
            for folder in os.listdir(patient_root):
                folder_display = folder.replace("_", " ")
                if folder_display.lower() in p_lower or folder.lower() in p_lower:
                    return folder_display
        return ""

    # ── Patient file finder ────────────────────────────────
    def _find_patient_file(self, patient_name: str, file_type: str = "any") -> str:
        """
        Walk the local patient directory and return the best-matching
        real file path using fuzzy name matching.

        patient_name : name spoken by doctor (may be partial/imperfect)
        file_type    : 'scan' | 'record' | 'any'
        """
        import difflib
        patient_root = os.path.join(V3_DIR, "data")
        if not os.path.isdir(patient_root):
            print(f"[v3] Patient directory not found: {patient_root}")
            return None

        SCAN_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"}
        RECORD_EXTS = {".pdf", ".txt", ".docx"}
        ALL_EXTS    = SCAN_EXTS | RECORD_EXTS

        name_query = patient_name.lower().replace("_", " ")
        best_score = 0.0
        best_path  = None

        for folder in os.listdir(patient_root):
            folder_path = os.path.join(patient_root, folder)
            if not os.path.isdir(folder_path):
                continue

            folder_clean = folder.replace("_", " ").lower()
            # Try substring match first (faster for exact names)
            if name_query in folder_clean or folder_clean in name_query:
                name_score = 0.95
            else:
                name_score = difflib.SequenceMatcher(None, name_query, folder_clean).ratio()

            if name_score < 0.4:
                continue  # Not this patient

            # Find the best file in this patient's folder
            for fname in os.listdir(folder_path):
                ext = os.path.splitext(fname)[1].lower()

                # Filter by requested file type
                if file_type in ("scan", "image") and ext not in SCAN_EXTS:
                    continue
                if file_type == "record" and ext not in RECORD_EXTS:
                    continue
                if ext not in ALL_EXTS:
                    continue

                # Score: prefer the file whose *stem* matches the requested type.
                # A file named "scan.jpg" scores +0.3 when file_type="scan",
                # guaranteeing it beats "Medical_Record.pdf" in the same folder.
                file_score = name_score
                fstem = os.path.splitext(fname)[0].lower()
                if file_type in ("scan", "image"):
                    if fstem in {"scan", "xray", "rontgen", "radiologi", "foto", "gambar"}:
                        file_score += 0.3  # explicit name match → always win
                    elif ext in SCAN_EXTS:
                        file_score += 0.1  # right extension but generic name
                if file_type == "record":
                    if fstem in {"medical_record", "rekam_medis", "record", "medical"}:
                        file_score += 0.3
                    elif ext in RECORD_EXTS:
                        file_score += 0.1

                if file_score > best_score:
                    best_score = file_score
                    best_path  = os.path.join(folder_path, fname)

        if best_path:
            print(f"[v1] Patient file found (score={best_score:.2f}): {best_path}")
        else:
            print(f"[v1] No patient file found for: {patient_name!r} type={file_type!r}")
        return best_path

    # ── App launcher ───────────────────────────────────────
    def _execute_launcher(self, action: str, file_path: str = None):
        """POST to host_bridge_v1 /run with validated action and optional real path."""
        payload = {"action": action}
        if file_path:
            payload["file"] = file_path
        print(f"[v1] Launching: action={action!r} file={file_path!r}")
        try:
            r = requests.post(HOST_BRIDGE_V1_URL, json=payload, timeout=5)
            if r.ok:
                label = f"{action}" + (f" ({os.path.basename(file_path)})" if file_path else "")
                self._add_chat("System", f"Launched: {label}", is_user=False)
            else:
                self._add_chat("System", f"Bridge error {r.status_code}", is_user=False)
        except Exception as e:
            self._add_chat("System", f"Bridge unreachable: {e}", is_user=False)

    # ── Screen control ─────────────────────────────────────
    def _execute_screen_control(self, params: dict):
        """POST to host_bridge_v1 /screen."""
        sc_action = params.get("action", "")
        region    = params.get("region", "center")
        times     = params.get("times", 1)
        print(f"[v1] Screen control: {sc_action} at {region} x{times}")
        try:
            r = requests.post(
                SCREEN_CONTROL_URL,
                json={"action": sc_action, "region": region, "times": times},
                timeout=5,
            )
            if r.ok:
                self._add_chat("System", f"Screen: {sc_action} @ {region} x{times}", is_user=False)
            else:
                self._add_chat("System", f"Screen error {r.status_code}", is_user=False)
        except Exception as e:
            self._add_chat("System", f"Screen control error: {e}", is_user=False)

    # ── Legacy unified dispatcher (kept for compatibility) ──
    def _execute_tool(self, tool: dict):
        name   = tool.get("name", "")
        params = tool.get("params", {})
        if name in ("windows_launcher_run", "windows-launcher-run"):
            self._execute_launcher(params.get("action", ""), params.get("path") or params.get("file"))
        elif name in ("screen_control", "screen-control"):
            self._execute_screen_control(params)
        else:
            self._add_chat("System", f"Unknown tool '{name}' - skipped.", is_user=False)

    def _generate_quick_feedback(self, text: str):
        try:
            system_prompt = (
                "Anda adalah Jarvis asisten AI. Pengguna baru saja memberikan perintah.\n"
                "Berikan respons konfirmasi SANGAT SINGKAT (maksimal 1 kalimat pendek) "
                "bahwa Anda akan memproses perintah tersebut.\n"
                "Jangan memberikan jawaban, hanya konfirmasi.\n"
                "Contoh: 'Baik, saya akan membuka data pasien Budi Setiawan.' atau 'Tunggu sebentar, saya proses.'"
            )
            r = requests.post("http://localhost:11434/api/generate", json={
                "model":  "llama3.2:latest",
                "prompt": text,
                "stream": False,
                "system": system_prompt,
            }, timeout=8)
            if r.ok:
                response_text = r.json().get("response", "").strip()
                self._tts_worker(response_text)
        except Exception:
            pass

    # ── Llama 3.2 voice formatter (+ RAG fallback) ─────────
    def _llm_tts_worker(self, text: str, already_processed: bool = False,
                        prefetched_rag: str = None):
        """
        already_processed=True  → text is Llama 3.1's answer; just polish it for voice.
        already_processed=False → raw user prompt; use prefetched_rag if available,
                                   otherwise run a fresh RAG query (fallback mode).
        prefetched_rag          → RAG context already fetched by _openclaw_pipeline;
                                   skips the redundant second query.
        """
        try:
            if already_processed:
                system_prompt = (
                    "Anda adalah Jarvis, asisten AI medis ProTel. "
                    "Ubah teks berikut menjadi respons suara yang singkat, alami, dan ramah "
                    "dalam bahasa Indonesia. Gunakan kalimat pendek. "
                    "Hindari tanda baca seperti *, #, atau simbol lainnya."
                )
                augmented = text
            else:
                # Use the pre-fetched RAG context if supplied; otherwise do a fresh query.
                if prefetched_rag:
                    rag_context = prefetched_rag
                    print(f"[v1] _llm_tts_worker using pre-fetched RAG ({len(rag_context)} chars)")
                else:
                    search_query = f"{text} rekam medis pasien"
                    rag_context  = emr_rag.query(search_query) if emr_rag.is_ready() else None
                    if rag_context:
                        print(f"[v1] _llm_tts_worker fallback RAG query returned {len(rag_context)} chars")

                if rag_context:
                    system_prompt = (
                        "Anda adalah Jarvis, asisten AI medis untuk dokter bedah ProTel. "
                        "Jawab pertanyaan dokter HANYA berdasarkan data rekam medis yang diberikan. "
                        "Jika informasi tidak ada dalam konteks, katakan bahwa data tidak tersedia. "
                        "Jawab singkat, padat, dan akurat dalam bahasa Indonesia."
                    )
                    augmented = f"Data Rekam Medis Pasien:\n{rag_context}\n\nPertanyaan Dokter: {text}"
                    self._add_chat("System", "📋 Menggunakan data rekam medis...", is_user=False)
                else:
                    system_prompt = (
                        "Anda adalah Jarvis, asisten AI medis untuk dokter bedah ProTel. "
                        "Jawab singkat, padat, dan profesional dalam bahasa Indonesia."
                    )
                    augmented = text

            r = requests.post("http://localhost:11434/api/generate", json={
                "model":  "llama3.2:latest",
                "prompt": augmented,
                "stream": False,
                "system": system_prompt,
            }, timeout=30)

            if r.ok:
                response_text = r.json().get("response", "").strip()
                self._add_chat("Jarvis", response_text, is_user=False)
                # If this wasn't already processed by Llama 3.1, store it in history
                if not already_processed:
                    self._session_history.append({"role": "assistant", "content": response_text})
                self._tts_worker(response_text)
            else:
                print(f"[v1] Llama 3.2 error: {r.status_code} {r.text}")
                self._add_chat("System", f"❌ Llama 3.2 error: {r.status_code}", is_user=False)

        except Exception as e:
            self._add_chat("System", f"❌ LLM/TTS Error: {e}", is_user=False)

    # ── TTS-only path ──────────────────────────────────────
    def _speak(self, text: str):
        threading.Thread(target=self._tts_worker, args=(text,), daemon=True).start()

    def _tts_worker(self, text: str):
        try:
            import uuid, re
            out_file = f"tts_v1_{uuid.uuid4().hex[:6]}.wav"
            tts_text = re.sub(r"[*+#_~`\[\]()]", "", text)
            from tts_piper import synthesize_indonesian
            synthesize_indonesian(tts_text, out_file)
            import winsound
            if os.path.exists(out_file):
                winsound.PlaySound(out_file, winsound.SND_FILENAME)
                os.remove(out_file)
        except Exception as e:
            self._add_chat("System", f"❌ TTS Error: {e}", is_user=False)

    # ── STATE MANAGEMENT ──────────────────────────────────────
    def _set_gest_wake(self, wake: bool):
        self._gest_wake = wake
        self._gest_thread.set_active(wake)
        self.update()
        if wake:
            self._g_badge.setText("G: WAKE")
            self._g_badge.setStyleSheet(
                "color: #44FF88; font-size: 11px; font-weight: bold;"
                " background: rgba(68,255,136,20); border: 1px solid rgba(68,255,136,60);"
                " border-radius: 8px; padding: 2px 8px;"
            )
        else:
            self._g_badge.setText("G: SLEEP")
            self._g_badge.setStyleSheet(
                "color: #FF4444; font-size: 11px; font-weight: bold;"
                " background: rgba(255,68,68,20); border: 1px solid rgba(255,68,68,60);"
                " border-radius: 8px; padding: 2px 8px;"
            )

    def _set_voice_wake(self, wake: bool):
        self._voice_wake = wake
        self._stt_thread.set_voice_awake(wake)
        self._stt_thread.set_force_listen(wake)
        self.update()
        if wake:
            self._v_badge.setText("V: WAKE")
            self._v_badge.setStyleSheet(
                "color: #44FF88; font-size: 11px; font-weight: bold;"
                " background: rgba(68,255,136,20); border: 1px solid rgba(68,255,136,60);"
                " border-radius: 8px; padding: 2px 8px;"
            )
        else:
            self._v_badge.setText("V: SLEEP")
            self._v_badge.setStyleSheet(
                "color: #FF4444; font-size: 11px; font-weight: bold;"
                " background: rgba(255,68,68,20); border: 1px solid rgba(255,68,68,60);"
                " border-radius: 8px; padding: 2px 8px;"
            )
            self._stt_overlay.update_state("STT: Waiting for 'Hello Zero Touch'...", "#00FFFF")

    # ── PTT ───────────────────────────────────────────────────
    def _on_ptt_pressed(self):
        self._ptt_btn.setStyleSheet(
            "QPushButton { background: #555555; color: #00FFFF; border-radius: 5px;"
            " font-weight: bold; margin: 5px 10px; }"
        )
        self._ptt_btn.setText("🎙️ Listening...")
        if hasattr(self, "_stt_thread"):
            self._stt_thread.ptt_start()

    def _on_ptt_released(self):
        self._ptt_btn.setStyleSheet(
            "QPushButton { background: #333333; color: white; border-radius: 5px;"
            " font-weight: bold; margin: 5px 10px; }"
        )
        self._ptt_btn.setText("Hold SPACE to Talk")
        self._last_was_ptt = True
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

    # ── CHAT LOG ──────────────────────────────────────────────
    def _add_chat(self, sender: str, text: str, is_user: bool):
        self.chat_signal.emit(sender, text, is_user)

    def _do_add_chat(self, sender: str, text: str, is_user: bool):
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

    # ── PAINTING ──────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0, QColor(18, 18, 22, 245))
        grad.setColorAt(1, QColor(10, 10, 14, 245))
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)
        if self._voice_wake or self._gest_wake:
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
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 20, 20)

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
        print("[v1] Shutting down…")
        if hasattr(self, "_bridge_proc") and self._bridge_proc:
            self._bridge_proc.terminate()
            try:
                self._bridge_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._bridge_proc.kill()
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
    font = QFont("Inter")
    if not font.exactMatch():
        font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = ZeroTouchV3()
    window.show()
    sys.exit(app.exec())
