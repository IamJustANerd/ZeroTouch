# run this natively on your Windows machine: python host_bridge.py
from flask import Flask, jsonify, request
import os
import subprocess

app = Flask(__name__)

# SAFE WHITELIST: Only these commands are allowed
# Mapping of keywords to actual Windows commands
ALLOWED_APPS = {
    "word": "start winword",
    "winword": "start winword",
    "whatsapp": "start whatsapp:",
    "notepad": "start notepad",
    "file": 'start ""'
}

@app.route('/run', methods=['POST'])
def run_action():
    data = request.json
    # AI might send 'action', 'app', or 'command'
    app_key = data.get("action") or data.get("app") or data.get("command")
    # AI might send 'file', 'path', or 'filepath'
    file_path = data.get("file") or data.get("path") or data.get("filepath")
    
    # If the action is "open", default to generic file opener, but override for specific types
    if app_key == "open" and file_path:
        app_key = "file"
        if file_path.endswith((".doc", ".docx")):
            app_key = "word"
        elif file_path.endswith(".txt"):
            app_key = "notepad"

    if app_key in ALLOWED_APPS:
        cmd = ALLOWED_APPS[app_key]
        
        if file_path:
            win_workspace = r"E:\OpenClawProject"
            container_workspace = "/home/node/.openclaw/workspace"
            
            if file_path.startswith(container_workspace):
                file_path = file_path.replace(container_workspace, win_workspace).replace("/", "\\")
            
            cmd = f'{cmd} "{file_path}"'
            
        print(f"Executing: {cmd}")
        subprocess.Popen(cmd, shell=True)
        return jsonify({"status": "success", "message": f"Executed: {cmd}"})
    
    return jsonify({"status": "error", "message": f"App '{app_key}' not allowed or recognized."}), 403

if __name__ == '__main__':
    print("Starting Improved Host Bridge on port 5000...")
    print(f"Allowed apps: {', '.join(ALLOWED_APPS.keys())}")
    app.run(host='0.0.0.0', port=5000)
