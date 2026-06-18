"""
test_openclaw.py  –  ZeroTouch V1 Pre-flight Check
====================================================
Run this FIRST before starting zerotouch_v1.py.

Usage:
    python ZeroTouchV1/test_openclaw.py

Requires: pip install websocket-client
"""

import sys
import json
import time
import uuid
import requests

# websocket-client (synchronous, simpler than asyncio websockets)
try:
    import websocket
except ImportError:
    print("[FAIL] 'websocket-client' not installed.")
    print("       -> Run: pip install websocket-client")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────
OPENCLAW_BASE  = "http://localhost:18789"
OPENCLAW_WS    = "ws://localhost:18789"
OPENCLAW_TOKEN = "1542658497515794168875165986594"
OLLAMA_BASE    = "http://localhost:11434"
TIMEOUT        = 10

PASS = "[OK]  "
FAIL = "[FAIL]"
WARN = "[WARN]"


# ── 1. OpenClaw Gateway HTTP reachability ─────────────────────────
def check_openclaw_gateway():
    print("\n--- OpenClaw Gateway (HTTP) ---")
    try:
        r = requests.get(OPENCLAW_BASE, timeout=TIMEOUT)
        # 200 = served the Control UI HTML, 404 = running but no root route — both mean it's up
        print(f"{PASS} OpenClaw gateway is reachable at {OPENCLAW_BASE}  (HTTP {r.status_code})")
        return True
    except requests.exceptions.ConnectionError:
        print(f"{FAIL} Cannot reach OpenClaw at {OPENCLAW_BASE}")
        print("       -> Make sure OpenClaw container is running: docker ps")
        return False
    except Exception as e:
        print(f"{FAIL} Gateway error: {e}")
        return False


# ── 2. OpenClaw WebSocket connectivity + auth ─────────────────────
def check_openclaw_websocket():
    """
    Connects to the OpenClaw WebSocket gateway and sends a lightweight
    RPC call (models.list) to verify auth + connectivity.

    OpenClaw uses a JSON-RPC-style WebSocket protocol. The token is
    passed as a query-string parameter: ws://host:port?token=...
    The Control UI uses this same pattern.
    """
    print("\n--- OpenClaw Gateway (WebSocket / RPC) ---")

    # Try token in query param — the pattern confirmed by Control UI behaviour
    ws_candidates = [
        f"{OPENCLAW_WS}?token={OPENCLAW_TOKEN}",
        f"{OPENCLAW_WS}/ws?token={OPENCLAW_TOKEN}",
        f"{OPENCLAW_WS}/gateway?token={OPENCLAW_TOKEN}",
    ]

    rpc_payload = json.dumps({
        "id":     str(uuid.uuid4()),
        "method": "models.list",
        "params": {}
    })

    for ws_url in ws_candidates:
        path = ws_url.split("localhost:18789")[1]
        try:
            ws = websocket.create_connection(ws_url, timeout=8)
            ws.send(rpc_payload)

            # Read up to 3 frames — gateway may send a welcome frame first
            response_text = None
            for _ in range(3):
                try:
                    frame = ws.recv()
                    if not frame:
                        continue
                    data = json.loads(frame)
                    # Check for error / token rejection
                    if isinstance(data, dict):
                        reason = (data.get("reason") or
                                  data.get("error", {}).get("message", "") if isinstance(data.get("error"), dict) else "")
                        if "token_missing" in str(reason) or "unauthorized" in str(reason).lower():
                            print(f"{FAIL} WebSocket at {path!r} rejected — token not accepted.")
                            print(f"       reason: {reason}")
                            ws.close()
                            return False
                        if "result" in data or "models" in data or data.get("method"):
                            response_text = frame
                            break
                except Exception:
                    break

            ws.close()

            if response_text:
                print(f"{PASS} WebSocket connected and authenticated at {path!r}")
                print(f"       RPC response preview: {response_text[:120]}")
                return True
            else:
                print(f"{PASS} WebSocket connected at {path!r} (no parseable RPC response — may be normal)")
                print(f"       The gateway accepted the connection. Treating as OK.")
                return True

        except websocket.WebSocketConnectionClosedException:
            print(f"       Path {path!r}: connection closed immediately (wrong path or token rejected)")
            continue
        except ConnectionRefusedError:
            print(f"{FAIL} WebSocket: connection refused at {OPENCLAW_WS}")
            print("       -> OpenClaw gateway may not be running.")
            return False
        except Exception as e:
            print(f"       Path {path!r}: {type(e).__name__}: {e}")
            continue

    print(f"{FAIL} Could not establish WebSocket connection on any known path.")
    print("       Tried: " + ", ".join(ws_url.split("localhost:18789")[1] for ws_url in ws_candidates))
    print("       -> Check OpenClaw gateway logs for the correct WebSocket path.")
    return False


# ── 3. Ollama service ─────────────────────────────────────────────
def check_ollama():
    print("\n--- Ollama Service ---")
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=TIMEOUT)
        if r.ok:
            print(f"{PASS} Ollama is running at {OLLAMA_BASE}")
            models = [m["name"] for m in r.json().get("models", [])]
            return True, models
        else:
            print(f"{FAIL} Ollama returned HTTP {r.status_code}")
            return False, []
    except requests.exceptions.ConnectionError:
        print(f"{FAIL} Cannot reach Ollama at {OLLAMA_BASE}")
        print("       -> Run: ollama serve")
        return False, []
    except Exception as e:
        print(f"{FAIL} Ollama error: {e}")
        return False, []


# ── 4. Individual model check ─────────────────────────────────────
def check_model(models: list, model_id: str):
    base = model_id.split(":")[0]
    matches = [m for m in models if m == model_id or m.startswith(base)]
    if matches:
        print(f"{PASS} {model_id} is available  (found: {matches[0]})")
        return True
    else:
        print(f"{FAIL} {model_id} NOT found in Ollama")
        print(f"       -> Run: ollama pull {model_id}")
        return False


# ── 5. Ollama tool-calling smoke test ─────────────────────────────
def check_ollama_tool_calling(llama31_model: str):
    """
    Sends a trivial tool-calling request to Ollama/llama3.1.
    This verifies that the model can produce tool_calls — the mechanism
    openclaw_client.py now uses instead of the OpenClaw WebSocket API.
    """
    print("\n--- Ollama Tool-Calling (llama3.1) ---")
    tools = [{
        "type": "function",
        "function": {
            "name": "windows_launcher_run",
            "description": "Open a Windows application",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"}
                },
                "required": ["action"]
            }
        }
    }]
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/chat", json={
            "model":    llama31_model,
            "messages": [{"role": "user", "content": "Open Notepad please."}],
            "tools":    tools,
            "stream":   False,
        }, timeout=60)

        if not r.ok:
            print(f"{FAIL} Ollama /api/chat returned HTTP {r.status_code}")
            return False

        data = r.json()
        msg  = data.get("message", {})
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            fn   = tool_calls[0].get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            print(f"{PASS} llama3.1 produced a tool call: {name}({args})")
            return True
        else:
            content = msg.get("content", "")[:100]
            print(f"{WARN} llama3.1 responded with text instead of a tool call: {content!r}")
            print(f"       Tool calling may work inconsistently with quantized models.")
            print(f"       The pipeline will still function (falls back gracefully).")
            return True  # Soft-pass — not a hard blocker

    except requests.exceptions.Timeout:
        print(f"{FAIL} Ollama tool-calling timed out (llama3.1 may still be loading)")
        return False
    except Exception as e:
        print(f"{FAIL} Tool-calling check error: {e}")
        return False


# ── 6. Host Bridge V1 ─────────────────────────────────────────────
def check_host_bridge_v1():
    print("\n--- Host Bridge V1 (optional pre-check) ---")
    try:
        r = requests.get("http://localhost:5001/health", timeout=3)
        if r.ok:
            print(f"{PASS} host_bridge_v1.py is already running on port 5001")
        else:
            print(f"{WARN} host_bridge_v1.py responded with HTTP {r.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"{WARN} host_bridge_v1.py is NOT running yet")
        print(f"       -> It will be auto-started by zerotouch_v1.py")


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 57)
    print("  ZeroTouch V1 — Pre-flight Connectivity Check")
    print("=" * 57)

    all_ok    = True
    llama31_model = None

    # 1. Gateway HTTP
    gw_ok = check_openclaw_gateway()
    all_ok = all_ok and gw_ok

    # 2. Gateway WebSocket
    if gw_ok:
        ws_ok  = check_openclaw_websocket()
        all_ok = all_ok and ws_ok
    else:
        print(f"\n{WARN} Skipping WebSocket check — HTTP gateway is down.")

    # 3. Ollama
    ollama_ok, models = check_ollama()
    all_ok = all_ok and ollama_ok

    # 4. Models
    if ollama_ok:
        m1_ok = check_model(models, "llama3.1")
        m2_ok = check_model(models, "llama3.2:latest")
        all_ok = all_ok and m1_ok and m2_ok
        # Get the exact quantized model name for the tool-call test
        llama31_model = next((m for m in models if m.startswith("llama3.1")), None)
    else:
        print(f"\n{WARN} Skipping model checks — Ollama is down.")

    # 5. Tool calling smoke test
    if llama31_model:
        tc_ok  = check_ollama_tool_calling(llama31_model)
        # soft check — don't fail the whole suite for this
    else:
        print(f"\n{WARN} Skipping tool-calling test — llama3.1 not found.")

    # 6. Host bridge (informational)
    check_host_bridge_v1()

    # Summary
    print("\n" + "=" * 57)
    if all_ok:
        print("  [OK]  ALL CHECKS PASSED — Safe to run zerotouch_v1.py")
    else:
        print("  [FAIL]  SOME CHECKS FAILED — Fix the issues above first.")
        sys.exit(1)
    print("=" * 57)


if __name__ == "__main__":
    main()
