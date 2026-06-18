"""
demo_routing.py  --  ZeroTouch V1 Routing Transparency Demo
=============================================================
Proves that Llama 3.1 is making the routing decision, NOT hardcoded rules.

How it works:
  - Sends your text directly to Ollama /api/chat with tool definitions
  - Prints the FULL RAW response from the model (nothing hidden)
  - Shows what action would be taken and WHY

Run:
    cd e:\\OpenclawMainFolder
    python ZeroTouchV1\\demo_routing.py

Try these commands to see the AI reason:
  "buka word"                             <- v0 also handles (keyword)
  "saya mau nulis catatan"                <- v0 CANNOT handle (no keyword)
  "zoom ke kanan atas 3 kali"             <- v0 CANNOT handle (new feature)
  "tolong tampilkan aplikasi chat keluarga"  <- v0 CANNOT handle (ambiguous)
  "siapa pasien hari ini?"                <- should return text, not a tool call
"""

import sys
import os
import json
import requests

# Add parent dir so we can import openclaw_client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openclaw_client

OLLAMA_BASE = "http://localhost:11434"
SEPARATOR   = "-" * 60


def run_demo(user_text: str):
    print(f"\n{SEPARATOR}")
    print(f"  USER INPUT: {user_text!r}")
    print(SEPARATOR)

    # ── Step 1: Show what we're sending to the model ──────────
    model_name = openclaw_client._resolve_model_name()
    payload = {
        "model":    model_name,
        "messages": [
            {"role": "system", "content": openclaw_client.SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        "tools":  openclaw_client.TOOLS,
        "stream": False,
    }

    print(f"\n[STEP 1] Sending to Ollama model: {model_name}")
    print(f"         System prompt (first 80 chars): {openclaw_client.SYSTEM_PROMPT[:80]}...")
    print(f"         Tools offered to the model:")
    for t in openclaw_client.TOOLS:
        fn = t["function"]
        print(f"           - {fn['name']}: {fn['description'][:60]}...")

    # ── Step 2: Get and show the FULL raw response ────────────
    print(f"\n[STEP 2] Waiting for Llama 3.1 response...")
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=90)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
        print(f"         ERROR: {e}")
        return

    print(f"\n[STEP 3] RAW response from Ollama (unmodified):")
    print(json.dumps(raw.get("message", {}), indent=4, ensure_ascii=False))

    # ── Step 3: Parse and explain the decision ────────────────
    result = openclaw_client._parse_ollama_response(raw)
    print(f"\n[STEP 4] Parsed decision:")
    print(f"         type       = {result.type!r}")
    print(f"         text       = {result.text!r}")
    print(f"         tool_calls = {result.tool_calls}")

    print(f"\n[STEP 5] What zerotouch_v1.py would do:")
    if result.type == "error":
        print(f"         Show error in chat UI: {result.text}")
    elif result.type == "tool_call":
        for tc in result.tool_calls:
            name   = tc["name"]
            params = tc["params"]
            if name == "windows_launcher_run":
                print(f"         -> Call host_bridge_v1 /run with: {params}")
                print(f"            (opens the app on Windows)")
            elif name == "screen_control":
                print(f"         -> Call host_bridge_v1 /screen with: {params}")
                print(f"            (controls the screen via pyautogui)")
            else:
                print(f"         -> Unknown tool {name!r} — would be skipped")
        if result.text:
            print(f"         -> Also send text to Llama 3.2 for TTS: {result.text!r}")
    else:
        print(f"         -> Send text to Llama 3.2 to polish, then speak via TTS:")
        print(f"            {result.text!r}")

    print(f"\n[PROOF]  The decision above was made by {model_name}, not by any")
    print(f"         keyword list. There is no 'if word in text' in this pipeline.")
    print(f"         The model read the tool definitions and chose what to call.")


def interactive_mode():
    print("=" * 60)
    print("  ZeroTouch V1 -- AI Routing Transparency Demo")
    print("  Proves Llama 3.1 is making decisions, not hardcoded rules")
    print("=" * 60)
    print("\nSuggested commands to try (Ctrl+C to quit):")
    suggestions = [
        "buka word",
        "saya mau nulis catatan",
        "tolong besarkan gambar di kanan atas 2 kali",
        "tampilkan aplikasi pesan untuk keluarga",
        "siapa pasien hari ini?",
        "buka file excel laporan",
        "zoom in top right 3x",
    ]
    for s in suggestions:
        print(f"  * {s}")
    print()

    while True:
        try:
            text = input("Enter command (or 'q' to quit): ").strip()
            if text.lower() in ("q", "quit", "exit"):
                break
            if not text:
                continue
            run_demo(text)
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # If a command is passed as argument, run it directly
    if len(sys.argv) > 1:
        run_demo(" ".join(sys.argv[1:]))
    else:
        interactive_mode()
