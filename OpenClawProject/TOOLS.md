# TOOLS.md - Local Notes

## Windows Host Bridge

You are inside a Linux Docker container. Your user is on a **Windows machine**.

**CRITICAL:** To open Windows apps, use the `windows-launcher-run` tool — NOT shell commands.

| App | Action keyword |
|---|---|
| Microsoft Word | `word` |
| Notepad | `notepad` |
| WhatsApp | `whatsapp` |

Example: `windows-launcher-run({"action": "word"})`

Never use `exec` for Windows apps. It will always fail.
