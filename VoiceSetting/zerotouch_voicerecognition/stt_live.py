"""
ProTel Live STT - Wake Word Mode (Fixed & Optimized)
=====================================================
"""

import os
import sys
import json
import queue
import logging
import warnings
import time
from datetime import datetime
from enum import Enum, auto

# ── Fix Windows terminal encoding ───────────────────────────────
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Suppress noisy HF Hub / TF warnings ─────────────────────────
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub.*")

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel

# ════════════════════════ CONFIG ═════════════════════════════════
MODEL_SIZE    = "small"       
DEVICE        = "cpu"         
COMPUTE_TYPE  = "int8"        
CPU_THREADS   = 16            
LANGUAGE      = "id"          

WAKE_WORD_MODEL     = "hey_jarvis"
WAKE_WORD_LABEL     = "hey jarvis"
# NAIKKAN THRESHOLD agar tidak sering bocor oleh noise ruangan
WAKE_WORD_THRESHOLD = 0.9             

SAMPLE_RATE = 16000   
CHANNELS    = 1       
OWW_CHUNK_SIZE = 1280

# TURUNKAN THRESHOLD agar deteksi suara manusiamu bisa mereset timer
SILENCE_THRESHOLD = 0.01    
SILENCE_TIMEOUT   = 3.5

OUTPUT_DIR = "text-from-stt"   
# ═════════════════════════════════════════════════════════════════


class State(Enum):
    IDLE      = auto()
    LISTENING = auto()

_audio_queue: queue.Queue = queue.Queue()


def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    _audio_queue.put(indata.copy())

def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

def _float32_to_int16(audio: np.ndarray) -> np.ndarray:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)

def _save_json(wake_word: str, confidence: float, transcription: str, duration_s: float, session_id: int) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now()
    filename = ts.strftime("%Y%m%d_%H%M%S") + f"_session{session_id:04d}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    payload = {
        "timestamp":        ts.isoformat(),
        "session_id":       session_id,
        "wake_word":        wake_word,
        "wake_word_confidence": round(confidence, 4),
        "transcription":    transcription,
        "language":         LANGUAGE,
        "stt_model":        MODEL_SIZE,
        "duration_seconds": round(duration_s, 2),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return filepath

def _print_banner(model_size: str) -> None:
    print("=" * 60)
    print("  ProTel Live STT -- Wake Word Mode")
    print("=" * 60)
    print(f"  STT Model   : {model_size}")
    print(f"  Wake Word   : \"{WAKE_WORD_LABEL}\"")
    print(f"  Language    : {LANGUAGE or 'auto-detect'}")
    print(f"  Silence Out : {SILENCE_TIMEOUT}s")
    print(f"  Output Dir  : {OUTPUT_DIR}/")
    print("=" * 60)

def main() -> None:
    _print_banner(MODEL_SIZE)

    print("\n⏳ Loading STT model (faster-whisper)...")
    stt_model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=CPU_THREADS,
    )
    print("✅ STT model ready.")

    print("⏳ Loading wake word model (openwakeword)...")
    oww_model = WakeWordModel(
        wakeword_models=[WAKE_WORD_MODEL],
        inference_framework="onnx",
    )
    print(f"✅ Wake word model ready.  Trigger → \"{WAKE_WORD_LABEL}\"")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    state            = State.IDLE
    session_id       = 0
    session_buffer   = []
    session_start    = 0.0
    last_speech_time = 0.0
    detected_conf    = 0.0
    session_max_rms  = 0.0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=OWW_CHUNK_SIZE,
        callback=_audio_callback,
    )

    print(f"\n[IDLE] Menunggu wake word -> \"{WAKE_WORD_LABEL}\"")
    print("-" * 60)

    try:
        with stream:
            while True:
                block = _audio_queue.get()
                mono  = block[:, 0]

                # ══════════════ STATE: IDLE ═══════════════════════
                if state == State.IDLE:
                    chunk_i16  = _float32_to_int16(mono)
                    predictions = oww_model.predict(chunk_i16)

                    for ww_name, score in predictions.items():
                        if score >= WAKE_WORD_THRESHOLD:
                            detected_conf    = float(score)
                            session_id      += 1
                            state            = State.LISTENING
                            session_buffer   = [mono.copy()]
                            session_start    = time.time()
                            last_speech_time = time.time()
                            session_max_rms  = _rms(mono)

                            print(
                                f"\n[DETECTED] Wake word terdeteksi!  "
                                f"(\"{WAKE_WORD_LABEL}\"  confidence: {detected_conf:.2f})"
                            )
                            print("[LISTENING] Silakan berbicara...")
                            print("-" * 60)
                            break

                # ══════════════ STATE: LISTENING ══════════════════
                elif state == State.LISTENING:
                    session_buffer.append(mono.copy())

                    current_rms = _rms(mono)
                    if current_rms > session_max_rms:
                        session_max_rms = current_rms

                    # Update last-speech timestamp jika audio memenuhi threshold bicara
                    if current_rms >= SILENCE_THRESHOLD:
                        last_speech_time = time.time()

                    silence_dur = time.time() - last_speech_time
                    elapsed     = time.time() - session_start

                    bar_filled  = int((silence_dur / SILENCE_TIMEOUT) * 20)
                    bar         = "#" * bar_filled + "." * (20 - bar_filled)
                    sys.stdout.write(
                        f"  [REC] Elapsed: {elapsed:4.1f}s  "
                        f"Silence: [{bar}] {silence_dur:.1f}/{SILENCE_TIMEOUT}s\r"
                    )
                    sys.stdout.flush()

                    # ── Silence timeout → transcribe & save ───────
                    if silence_dur >= SILENCE_TIMEOUT:
                        print(f"\n[PAUSE] Silence {silence_dur:.1f}s terdeteksi -- memproses...")
                        print(f"  [DEBUG] Peak audio RMS dalam sesi ini: {session_max_rms:.4f} (Threshold: {SILENCE_THRESHOLD})")

                        audio_data   = np.concatenate(session_buffer)
                        total_dur    = time.time() - session_start

                        # NORMALIZATION BLOCK DIHAPUS - Mencegah halusinasi Whisper

                        segments, _info = stt_model.transcribe(
                            audio_data,
                            beam_size=5,
                            language=LANGUAGE,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=400),
                            condition_on_previous_text=True,
                            initial_prompt=(
                                "Berikut adalah percakapan dalam bahasa Indonesia, kata yang dibicarakan adalah kata formal dan informal, pastikan hasil transkripsi adalah kata yang valid."
                            ),
                        )

                        text_parts    = [seg.text.strip() for seg in segments]
                        transcription = " ".join(text_parts).strip()

                        if transcription:
                            print(f"\n  [TEXT] Transkripsi : {transcription}")
                        else:
                            print("\n  [TEXT] Transkripsi : (tidak ada suara terdeteksi)")

                        out_path = _save_json(
                            wake_word    = WAKE_WORD_LABEL,
                            confidence   = detected_conf,
                            transcription= transcription,
                            duration_s   = total_dur,
                            session_id   = session_id,
                        )
                        print(f"  [SAVED] Disimpan ke : {out_path}")

                        # ── Reset and go back to IDLE ─────────────
                        session_buffer = []
                        state          = State.IDLE
                        detected_conf  = 0.0
                        session_max_rms = 0.0  # FIX: Reset RMS maximum untuk sesi selanjutnya

                        # FIX 1: Reset model openwakeword agar sisa audio dari sesi sebelumnya terhapus
                        if hasattr(oww_model, 'reset'):
                            oww_model.reset()
                            
                        # FIX 2: Kosongkan queue audio yang menumpuk selama proses transkripsi
                        with _audio_queue.mutex:
                            _audio_queue.queue.clear()

                        print(f"\n[IDLE] Menunggu wake word -> \"{WAKE_WORD_LABEL}\"")
                        print("-" * 60)

    except KeyboardInterrupt:
        print("\n" + "-" * 60)
        print("Dihentikan. Membersihkan...")
        print("Selesai!")

if __name__ == "__main__":
    main()