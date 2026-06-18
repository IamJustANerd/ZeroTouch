import websocket
import json
import uuid
import time

WS_URL = "ws://localhost:18789/ws?token=1542658497515794168875165986594"

def test_openclaw():
    ws = websocket.create_connection(WS_URL)
    
    # Send chat using jsonrpc 2.0
    req_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    
    req = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "chat.send",
        "params": {
            "sessionKey": "main",
            "message": "Halo, tolong catat 'Berhasil dari JSONRPC' ke success_rpc.txt",
            "idempotencyKey": run_id
        }
    }
    ws.send(json.dumps(req))
    
    while True:
        resp = json.loads(ws.recv())
        print(f"Resp: {resp}")
        if resp.get("id") == req_id:
            break
            
    ws.close()

if __name__ == "__main__":
    test_openclaw()
