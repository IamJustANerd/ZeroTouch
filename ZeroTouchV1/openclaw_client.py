"""
openclaw_client.py  --  ZeroTouch V1
=====================================
Sends voice commands to Llama 3.1 via Ollama's native tool-calling API.

Tools exposed to Llama 3.1:
  - rag_search           : Query the local EMR database (read patient data aloud)
  - find_patient_file    : Locate an actual file (scan/record) in the patient directory
  - windows_launcher_run : Open an app or file on Windows
  - screen_control       : Zoom/scroll/click at screen regions

Multi-turn flow for RAG queries:
  User prompt -> Llama 3.1 calls rag_search -> we execute and send results back
  -> Llama 3.1 writes a spoken answer -> Llama 3.2 formats -> TTS
"""

import json
import logging
import requests

logger = logging.getLogger(__name__)

OLLAMA_BASE   = "http://localhost:11434"
LLAMA31_MODEL = "llama3.1"
TIMEOUT       = 60

SYSTEM_PROMPT = (
    "Anda adalah Jarvis, asisten AI medis untuk dokter bedah ProTel. "
    "Pilih tool yang TEPAT berdasarkan permintaan dokter:\n\n"
    "- Dokter meminta MEMBACA / MERANGKUM / MENJELASKAN data pasien "
    "  (contoh: 'bacakan data Agus', 'ceritakan kondisi pasien') "
    "  -> Gunakan rag_search untuk mencari rekam medis, lalu jawab dalam teks.\n\n"
    "- Dokter meminta MEMBUKA FILE SCAN / GAMBAR / DOKUMEN pasien "
    "  (contoh: 'buka scan Agus', 'tampilkan hasil rontgen') "
    "  -> Gunakan find_patient_file untuk mendapatkan path file yang benar.\n\n"
    "- Dokter meminta MEMBUKA APLIKASI (word, excel, whatsapp, notepad, dll) "
    "  -> Gunakan windows_launcher_run.\n\n"
    "- Dokter meminta KONTROL LAYAR (zoom, scroll, klik) "
    "  -> Gunakan screen_control.\n\n"
    "- Pertanyaan umum yang tidak butuh tool "
    "  -> Jawab langsung dalam teks singkat bahasa Indonesia.\n\n"
    "PENTING: JANGAN menebak atau mengarang path file. "
    "Selalu gunakan find_patient_file untuk mendapatkan path yang sesungguhnya."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Cari dan ambil data rekam medis pasien dari database lokal. "
                "Gunakan ini ketika dokter meminta untuk membaca, merangkum, "
                "atau mendapatkan informasi tentang kondisi/riwayat pasien. "
                "Hasilnya akan dikembalikan ke Anda untuk dirangkum menjadi jawaban suara."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Query pencarian rekam medis. "
                            "Sertakan nama pasien dan informasi yang dicari. "
                            "Contoh: 'data lengkap Agus Setiawan' atau "
                            "'riwayat alergi Budi Santoso'"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_patient_file",
            "description": (
                "Cari file pasien (scan, gambar, rekam medis PDF) di direktori lokal. "
                "Gunakan ini ketika dokter meminta membuka gambar scan atau file pasien. "
                "Tool ini akan menemukan path file yang SESUNGGUHNYA. "
                "Jangan gunakan windows_launcher_run dengan path yang Anda tebak sendiri."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name": {
                        "type": "string",
                        "description": "Nama pasien, contoh: 'Agus Setiawan'",
                    },
                    "file_type": {
                        "type": "string",
                        "description": (
                            "Tipe file yang dicari: "
                            "'scan' (gambar scan/rontgen), "
                            "'record' (PDF rekam medis), "
                            "'any' (semua jenis file). "
                            "Default: 'any'"
                        ),
                        "default": "any",
                    },
                },
                "required": ["patient_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "windows_launcher_run",
            "description": (
                "Buka aplikasi Windows. "
                "Gunakan HANYA untuk membuka aplikasi umum (bukan file pasien). "
                "Untuk file pasien, gunakan find_patient_file terlebih dahulu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "Nama aplikasi. Nilai yang diizinkan: "
                            "word, excel, powerpoint, notepad, "
                            "explorer, photos, chrome, edge"
                        ),
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "screen_control",
            "description": (
                "Kontrol layar: zoom in/out, scroll, gerak mouse, atau klik. "
                "Gunakan untuk perintah seperti 'besarkan gambar kanan atas 2x'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": (
                            "Aksi yang dilakukan: "
                            "zoom-in, zoom-out, scroll-zoom-in, scroll-zoom-out, "
                            "scroll-up, scroll-down, move, click, double-click"
                        ),
                    },
                    "region": {
                        "type": "string",
                        "description": (
                            "Nama area layar: "
                            "top-left, top-center, top-right, "
                            "mid-left, center, mid-right, "
                            "bot-left, bot-center, bot-right"
                        ),
                        "default": "center",
                    },
                    "times": {
                        "type": "integer",
                        "description": "Berapa kali diulang. Default: 1",
                        "default": 1,
                    },
                },
                "required": ["action"],
            },
        },
    },
]


# ── Result object ─────────────────────────────────────────────────
class OpenClawResult:
    def __init__(self, result_type: str, text: str = "", tool_calls: list = None):
        self.type       = result_type
        self.text       = text.strip() if text else ""
        self.tool_calls = tool_calls or []

    def __repr__(self):
        snippet = self.text[:60].replace("\n", " ")
        return (
            f"OpenClawResult(type={self.type!r}, "
            f"text={snippet!r}, "
            f"tools={self.tool_calls})"
        )


# ── Public API ────────────────────────────────────────────────────
def is_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if not r.ok:
            return False
        models = [m["name"] for m in r.json().get("models", [])]
        return any(m.startswith(LLAMA31_MODEL) for m in models)
    except Exception:
        return False


def build_initial_messages(user_text: str) -> list:
    """Build the starting message list for a conversation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_text},
    ]


def append_tool_result(messages: list, tool_name: str, result_text: str) -> list:
    """
    Append a tool result to a conversation so Llama 3.1 can
    formulate its final answer based on the retrieved data.
    """
    return messages + [{"role": "tool", "content": result_text, "name": tool_name}]


def send_messages(messages: list) -> OpenClawResult:
    """
    Send a message list to Llama 3.1. Used for both the initial call
    and follow-up calls after tool results.
    """
    model_name = _resolve_model_name()
    payload = {
        "model":    model_name,
        "messages": messages,
        "tools":    TOOLS,
        "stream":   False,
    }
    logger.debug(f"[openclaw_client] Sending {len(messages)} messages to {model_name}")
    try:
        r = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        return OpenClawResult("error", text=f"Ollama not reachable at {OLLAMA_BASE}.")
    except requests.exceptions.Timeout:
        return OpenClawResult("error", text=f"Ollama timed out after {TIMEOUT}s.")
    except Exception as e:
        return OpenClawResult("error", text=str(e))

    if not r.ok:
        return OpenClawResult("error", text=f"Ollama HTTP {r.status_code}: {r.text[:200]}")

    try:
        data = r.json()
    except Exception as e:
        return OpenClawResult("error", text=f"Parse error: {e}")

    logger.debug(f"[openclaw_client] Response: {json.dumps(data.get('message', {}))[:300]}")
    return _parse_ollama_response(data)


def send_message(user_text: str) -> OpenClawResult:
    """Convenience wrapper: single-turn send."""
    return send_messages(build_initial_messages(user_text))


# ── Internal helpers ──────────────────────────────────────────────
def _resolve_model_name() -> str:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.ok:
            models = [m["name"] for m in r.json().get("models", [])]
            match  = next((m for m in models if m.startswith(LLAMA31_MODEL)), None)
            if match:
                return match
    except Exception:
        pass
    return LLAMA31_MODEL


def _parse_ollama_response(data: dict) -> OpenClawResult:
    msg       = data.get("message", {})
    content   = msg.get("content", "").strip()
    raw_tools = msg.get("tool_calls", [])

    tool_calls = []
    for tc in raw_tools:
        fn   = tc.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name:
            tool_calls.append({"name": name, "params": args})

    if tool_calls:
        return OpenClawResult("tool_call", text=content, tool_calls=tool_calls)
    return OpenClawResult("text", text=content)
