import os, random
from tts_piper import synthesize_indonesian

PHRASES       = ["Hello Zero Touch", "Hello, Zero Touch"]
OUTPUT_DIR    = "training_data/positive"
NUM_SAMPLES   = 1000
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parameter ranges — vary these to simulate speaker diversity
LENGTH_SCALES = [0.75, 0.85, 0.90, 0.95, 1.0, 1.05, 1.15]
NOISE_SCALES  = [0.5, 0.6, 0.667, 0.75, 0.8, 0.85, 0.9]
NOISE_WS      = [0.6, 0.7, 0.8, 0.85, 0.9, 1.0]

for i in range(NUM_SAMPLES):
    phrase = random.choice(PHRASES)
    out = os.path.join(OUTPUT_DIR, f"pos_{i:04d}.wav")
    synthesize_indonesian(
        phrase, out,
        length_scale = random.choice(LENGTH_SCALES),
        noise_scale  = random.choice(NOISE_SCALES),
        noise_w      = random.choice(NOISE_WS),
    )
    if i % 50 == 0:
        print(f"  Generated {i}/{NUM_SAMPLES}")

print("✅ Positive samples done.")