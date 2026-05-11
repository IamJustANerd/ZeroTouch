# OpenClaw Local Development Setup

This guide outlines the foolproof initialization and pairing process for running the OpenClaw AI Agent Gateway via Docker on Windows.

## 1. Directory Structure & Environment
Ensure your directories are set up on your host machine before starting.
* **Workspace:** `E:\OpenClawProject` (Where the AI will read/write files)
* **Config:** `E:\OpenClawConfig` (Where OpenClaw saves its internal database)

In your OpenClaw installation folder, create/edit the `.env` file with these exact three lines to enable Local Mode and bind the directories:
```env
OPENCLAW_CONFIG_DIR=E:\OpenClawConfig
OPENCLAW_WORKSPACE_DIR=E:\OpenClawProject
OPENCLAW_GATEWAY_MODE=local

## 2. Boot the Containers
Open a terminal in your OpenClaw installation folder and start the Docker containers natively:

DOS
docker compose up -d --pull never
Wait ~10 seconds for the Gateway to fully initialize and generate its internal security keys.

## 3. The Pairing Handshake
Because OpenClaw strictly enforces zero-trust device pairing, you must manually authorize your browser through the CLI.

Step A: Get the Active Token
Retrieve the current session token from the Gateway:

DOS
docker exec -it openclaw-openclaw-gateway-1 node dist/index.js dashboard
Copy only the token string located at the end of the URL (after #token=).

Step B: Trigger the Request

Open your browser and navigate to http://localhost:18789.

Ensure WebSocket URL is ws://localhost:18789.

Paste the token into the Gateway Token box. Leave the password blank.

Click Connect.
(You will receive a red "pairing required" error. This is expected—it just created a pending authorization request).

Step C: Approve the Device
Immediately check the terminal for the pending request:

DOS
docker exec -it openclaw-openclaw-gateway-1 node dist/index.js devices list
Find your browser under the Pending section and copy its Request ID. Then, approve it:

DOS
docker exec -it openclaw-openclaw-gateway-1 node dist/index.js devices approve <REQUEST_ID>
Step D: Final Connection
Return to your browser (the token should still be in the box) and click Connect one last time. The dashboard will unlock instantly.


Now that your infrastructure is rock-solid and the AI brain is officially online,