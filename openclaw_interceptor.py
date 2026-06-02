"""
openclaw_interceptor.py

Watches the OpenClaw session file for AI messages that contain a 
windows-launcher-run JSON call in plain text (Llama tool-call degradation),
and executes the corresponding Windows command directly.

Run this alongside host_bridge.py:
    python openclaw_interceptor.py
"""

import json
import re
import time
import glob
import subprocess
import os
import sys

# Flush output immediately so it shows in terminal
sys.stdout.reconfigure(line_buffering=True)

SESSION_DIR = r"E:\OpenclawMainFolder\OpenClawConfig\agents\main\sessions"

ALLOWED_APPS = {
    "word":     "start winword",
    "winword":  "start winword",
    "whatsapp": "start whatsapp:",
    "notepad":  "start notepad",
    "photos":   "start ms-photos:",
    "explorer": "start explorer",
}

# IDs of messages we have already acted on
processed_ids = set()


def find_latest_session():
    """Return the most recently modified .jsonl session file."""
    files = glob.glob(os.path.join(SESSION_DIR, "*.jsonl"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_launcher_call(text):
    """
    Try to extract (action, path) from a windows-launcher-run JSON string.
    Returns (action, path) or (None, None).
    """
    text = text.strip()

    # Strategy 1: full JSON parse
    try:
        data = json.loads(text)
        if data.get("name") == "windows-launcher-run":
            params = data.get("parameters", {})
            return params.get("action"), params.get("path")
    except Exception:
        pass

    # Strategy 2: regex (handles partial / malformed JSON)
    action_match = re.search(r'"(?:name)"\s*:\s*"windows-launcher-run"', text)
    if action_match:
        action = None
        path = None
        a = re.search(r'"action"\s*:\s*"([^"]+)"', text)
        p = re.search(r'"path"\s*:\s*"([^"]+)"', text)
        if a:
            action = a.group(1)
        if p:
            path = p.group(1)
        return action, path

    return None, None


def resolve_path(path):
    """Convert Linux container path to Windows path."""
    if not path:
        return None
    container_workspace = "/home/node/.openclaw/workspace"
    win_workspace = r"E:\OpenClawProject"
    if path.startswith(container_workspace):
        path = path.replace(container_workspace, win_workspace).replace("/", "\\")
    return path


def execute_action(action, path=None):
    """Map action keyword to a Windows command and run it."""
    # Smart extension-based fallback for 'open'
    if action == "open" and path:
        if path.endswith((".doc", ".docx")):
            action = "word"
        elif path.endswith(".txt"):
            action = "notepad"

    if action not in ALLOWED_APPS:
        print(f"[interceptor] Unknown action '{action}' — skipping.")
        return False

    cmd = ALLOWED_APPS[action]
    win_path = resolve_path(path)
    if win_path:
        cmd = f'{cmd} "{win_path}"'

    print(f"[interceptor] >> Executing: {cmd}")
    subprocess.Popen(cmd, shell=True)
    return True


def process_session(filepath):
    """Scan session file for unprocessed AI messages with launcher calls."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"[interceptor] Cannot read session file: {e}")
        return

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue

        if entry.get("type") != "message":
            continue

        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue

        msg_id = entry.get("id")
        if not msg_id or msg_id in processed_ids:
            continue

        # Check all text content parts
        for part in msg.get("content", []):
            if part.get("type") != "text":
                continue
            text = part.get("text", "")
            if "windows-launcher-run" not in text:
                continue

            action, path = parse_launcher_call(text)
            if action:
                print(f"[interceptor] FOUND: action={action!r} path={path!r}")
                execute_action(action, path)
                processed_ids.add(msg_id)
                break  # only act once per message


def main():
    print("=" * 55)
    print("  OpenClaw Windows Launcher Interceptor")
    print("  Watching for windows-launcher-run tool calls...")
    print(f"  Session dir: {SESSION_DIR}")
    print("=" * 55)

    last_session = None

    while True:
        session_file = find_latest_session()

        if session_file != last_session:
            if session_file:
                print(f"[interceptor] Watching session: {os.path.basename(session_file)}")
            last_session = session_file
            processed_ids.clear()  # new session = fresh start

        if session_file:
            process_session(session_file)

        time.sleep(0.8)


if __name__ == "__main__":
    main()
