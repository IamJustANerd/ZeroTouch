import os

file_path = r"C:\Users\User\Documents\AntiGravity\ProTel\venv\Lib\site-packages\openwakeword\data.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Patch torchaudio.load(clip) to use soundfile explicitly
content = content.replace(
    "torchaudio.load(clip)",
    'torchaudio.load(clip, backend="soundfile")'
)

# Patch torchaudio.load(i) to use soundfile explicitly
content = content.replace(
    "torchaudio.load(i)",
    'torchaudio.load(i, backend="soundfile")'
)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patched openwakeword.data.py to use soundfile backend.")
