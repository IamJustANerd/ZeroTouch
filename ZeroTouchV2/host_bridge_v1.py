"""
host_bridge_v1.py  –  ZeroTouch V1
=====================================
Extended host bridge running on port 5001 (v0 uses 5000, no conflict).

Endpoints
---------
POST /run     – Launch Windows apps (word, notepad, whatsapp, explorer, etc.)
POST /screen  – Advanced screen control (zoom-in/out, mouse move, scroll, click)
POST /rag     – Query the local EMR RAG index (for Llama 3.1 tool calls)
GET  /health  – Simple liveness probe

Screen Control Regions
-----------------------
The /screen endpoint maps named regions to screen percentage coordinates:
  top-left (25%, 25%)    top-center (50%, 25%)    top-right (75%, 25%)
  mid-left (25%, 50%)    center     (50%, 50%)    mid-right (75%, 50%)
  bot-left (25%, 75%)    bot-center (50%, 75%)    bot-right (75%, 75%)

Usage
-----
    python ZeroTouchV1/host_bridge_v1.py

It is auto-started by zerotouch_v1.py so you rarely need to run it manually.
"""

import os
import sys
import subprocess
import time

from flask import Flask, jsonify, request

# Allow importing emr_rag from the parent directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

app = Flask(__name__)

# ── App Whitelist ─────────────────────────────────────────────────
ALLOWED_APPS = {
    "word":      "start winword",
    "winword":   "start winword",
    "excel":     "start excel",
    "powerpoint":"start powerpnt",
    "notepad":   "start notepad",
    "explorer":  "start explorer",
    "photos":    "start ms-photos:",
    "chrome":    "start chrome",
    "edge":      "start msedge",
    "file":      'start ""',
}

# ── Screen Region Map (relative to screen size) ───────────────────
REGION_MAP = {
    "top-left":   (0.25, 0.25),
    "top-center": (0.50, 0.25),
    "top-right":  (0.75, 0.25),
    "mid-left":   (0.25, 0.50),
    "center":     (0.50, 0.50),
    "mid-right":  (0.75, 0.50),
    "bot-left":   (0.25, 0.75),
    "bot-center": (0.50, 0.75),
    "bot-right":  (0.75, 0.75),
    # Natural-language aliases
    "top right":        (0.75, 0.25),
    "top left":         (0.25, 0.25),
    "bottom right":     (0.75, 0.75),
    "bottom left":      (0.25, 0.75),
    "upper right":      (0.75, 0.25),
    "upper left":       (0.25, 0.25),
    "lower right":      (0.75, 0.75),
    "lower left":       (0.25, 0.75),
}


# ── /health ───────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "host_bridge_v1", "port": 5001})


# ── /run ──────────────────────────────────────────────────────────
@app.route("/run", methods=["POST"])
def run_action():
    """Launch a whitelisted Windows application or open a specific file."""
    data      = request.json or {}
    app_key   = data.get("action") or data.get("app") or data.get("command")
    file_path = data.get("file")  or data.get("path") or data.get("filepath")

    # Smart extension-based fallback for generic "open" action
    if app_key == "open" and file_path:
        ext = os.path.splitext(file_path)[1].lower()
        app_key = {
            ".doc": "word", ".docx": "word",
            ".xls": "excel", ".xlsx": "excel",
            ".ppt": "powerpoint", ".pptx": "powerpoint",
            ".txt": "notepad",
        }.get(ext, "file")

    if app_key not in ALLOWED_APPS:
        return jsonify({
            "status": "error",
            "message": f"App '{app_key}' is not in the whitelist.",
            "allowed": list(ALLOWED_APPS.keys()),
        }), 403

    # ── Special case: open a specific file with its default Windows app ──
    # os.startfile() is the correct Windows API for this.
    # DO NOT use "start ms-photos: path" — the URI protocol ignores file args.
    if app_key in ("file", "photos") and file_path:
        file_path = os.path.normpath(file_path)  # Fix any mixed slash issues
        if not os.path.exists(file_path):
            return jsonify({
                "status": "error",
                "message": f"File not found: {file_path}",
            }), 404
        print(f"[bridge_v1 /run] Opening file: {file_path}")
        try:
            os.startfile(file_path)   # Opens with default associated app
            return jsonify({"status": "success", "message": f"Opened: {file_path}"})
        except Exception as e:
            # Fallback to shell command
            subprocess.Popen(f'start "" "{file_path}"', shell=True)
            return jsonify({"status": "success", "message": f"Opened (shell): {file_path}"})

    # ── Standard app launch ───────────────────────────────────────────
    cmd = ALLOWED_APPS[app_key]

    if file_path:
        # Translate Linux container paths -> Windows paths
        container_ws = "/home/node/.openclaw/workspace"
        windows_ws   = r"E:\OpenClawProject"
        if file_path.startswith(container_ws):
            file_path = file_path.replace(container_ws, windows_ws).replace("/", "\\")
        cmd = f'{cmd} "{file_path}"'

    print(f"[bridge_v1 /run] Executing: {cmd}")
    subprocess.Popen(cmd, shell=True)
    return jsonify({"status": "success", "message": f"Executed: {cmd}"})



# ── /screen ───────────────────────────────────────────────────────
@app.route("/screen", methods=["POST"])
def screen_control():
    """
    Advanced screen manipulation via pyautogui.

    Body params:
        action : str  — "zoom-in" | "zoom-out" | "scroll-up" | "scroll-down"
                        | "move" | "click" | "double-click"
        region : str  — named region (see REGION_MAP) or "x,y" pixel coords
        times  : int  — repetitions / scroll clicks (default 1)
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = False   # Disable corner-failsafe for voice control
    except ImportError:
        return jsonify({
            "status": "error",
            "message": "pyautogui is not installed. Run: pip install pyautogui"
        }), 500

    data   = request.json or {}
    action = data.get("action", "").lower().replace("_", "-")
    region = str(data.get("region", "center")).lower().strip()
    times  = max(1, int(data.get("times", 1)))

    # Resolve region → pixel coordinates
    screen_w, screen_h = pyautogui.size()

    if "," in region:
        # Direct pixel input: "960,540"
        try:
            px, py = [int(v.strip()) for v in region.split(",")]
        except ValueError:
            return jsonify({"status": "error", "message": f"Invalid coords: {region}"}), 400
    else:
        rx, ry = REGION_MAP.get(region, (0.50, 0.50))
        px, py = int(screen_w * rx), int(screen_h * ry)

    print(f"[bridge_v1 /screen] action={action!r} region={region!r} → ({px},{py}) times={times}")

    try:
        if action == "zoom-in":
            pyautogui.moveTo(px, py, duration=0.15)
            for _ in range(times):
                pyautogui.hotkey("ctrl", "+")
                time.sleep(0.08)

        elif action == "zoom-out":
            pyautogui.moveTo(px, py, duration=0.15)
            for _ in range(times):
                pyautogui.hotkey("ctrl", "-")
                time.sleep(0.08)

        elif action == "scroll-zoom-in":
            # Ctrl + scroll-up: more compatible across image viewers
            pyautogui.moveTo(px, py, duration=0.15)
            pyautogui.keyDown("ctrl")
            pyautogui.scroll(times * 3, x=px, y=py)
            pyautogui.keyUp("ctrl")

        elif action == "scroll-zoom-out":
            pyautogui.moveTo(px, py, duration=0.15)
            pyautogui.keyDown("ctrl")
            pyautogui.scroll(-times * 3, x=px, y=py)
            pyautogui.keyUp("ctrl")

        elif action == "scroll-up":
            pyautogui.moveTo(px, py, duration=0.15)
            pyautogui.scroll(times * 3, x=px, y=py)

        elif action == "scroll-down":
            pyautogui.moveTo(px, py, duration=0.15)
            pyautogui.scroll(-times * 3, x=px, y=py)

        elif action == "move":
            pyautogui.moveTo(px, py, duration=0.3)

        elif action == "click":
            pyautogui.click(px, py)

        elif action == "double-click":
            pyautogui.doubleClick(px, py)

        else:
            return jsonify({
                "status": "error",
                "message": f"Unknown screen action: {action!r}",
                "supported": ["zoom-in", "zoom-out", "scroll-zoom-in", "scroll-zoom-out",
                              "scroll-up", "scroll-down", "move", "click", "double-click"],
            }), 400

        return jsonify({
            "status": "success",
            "message": f"Performed '{action}' at region='{region}' ({px},{py}) x{times}",
        })

    except Exception as e:
        print(f"[bridge_v1 /screen] Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ── /rag ──────────────────────────────────────────────────────────
@app.route("/rag", methods=["POST"])
def rag_query():
    """
    Expose the local EMR RAG index over HTTP.
    Called by zerotouch_v1.py directly, or potentially by Llama 3.1 as a tool.
    """
    try:
        import emr_rag
    except ImportError:
        return jsonify({"status": "error", "message": "emr_rag module not found"}), 500

    data     = request.json or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"status": "error", "message": "No 'question' field provided"}), 400

    if not emr_rag.is_ready():
        return jsonify({
            "status": "not_ready",
            "result": "RAG index is still loading. Please try again in a moment.",
        })

    result = emr_rag.query(question)
    if result:
        return jsonify({"status": "success", "result": result})
    return jsonify({"status": "success", "result": None,
                    "message": "No relevant patient records found."})


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  ZeroTouch V1  —  Host Bridge")
    print(f"  Listening on  http://0.0.0.0:5001")
    print(f"  Allowed apps: {', '.join(ALLOWED_APPS.keys())}")
    print(f"  Endpoints:    /run  /screen  /rag  /health")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5001, debug=False)
