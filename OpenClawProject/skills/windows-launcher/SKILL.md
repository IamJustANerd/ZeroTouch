# Windows Launcher

> **CRITICAL:** You are in a Linux container. You CANNOT run Windows apps via shell commands.
> Shell commands like `exec("microsoft-word")` or `exec("start winword")` will ALWAYS fail.
> This tool is the ONLY way to open apps on the Windows host machine.

## When to Use

Use `windows-launcher-run` ANY TIME the user asks you to:
- Open an application (Word, Notepad, WhatsApp, Photos, etc.)
- Open a file (a .docx, .txt, or any document)
- Launch any Windows program

## Tool Name
`windows-launcher-run`

## Parameters
- `action` (string, required): The app keyword. Supported values:
  - `word` — Opens Microsoft Word
  - `winword` — Same as `word`
  - `notepad` — Opens Notepad
  - `whatsapp` — Opens WhatsApp
  - `open` — Opens a specific file (use with `path`)
- `path` (string, optional): Full container path to a file to open.

## Examples

| User Request | Correct Tool Call |
|---|---|
| "Open Word" | `{"action": "word"}` |
| "Open Microsoft Word" | `{"action": "word"}` |
| "Open Notepad" | `{"action": "notepad"}` |
| "Open WhatsApp" | `{"action": "whatsapp"}` |
| "Open this file: Test.docx" | `{"action": "open", "path": "/home/node/.openclaw/workspace/Test.docx"}` |

## What NOT to Do

❌ `exec("microsoft-word")` — command not found in Linux  
❌ `exec("start winword")` — shell start won't work in container  
❌ Any other shell approach — this is a Windows host, not Linux  
