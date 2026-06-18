import os, shutil, random, glob

ESC50_DIR  = "ESC-50/audio"
OUTPUT_DIR = "training_data/negative"
NUM_NEG    = 3000
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_clips = glob.glob(os.path.join(ESC50_DIR, "*.wav"))
selected  = random.choices(all_clips, k=NUM_NEG)

for i, src in enumerate(selected):
    dst = os.path.join(OUTPUT_DIR, f"neg_{i:04d}.wav")
    shutil.copy(src, dst)

print(f"✅ {NUM_NEG} negative samples staged.")