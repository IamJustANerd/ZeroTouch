import os, glob, random, soundfile as sf, numpy as np
import warnings
from audiomentations import Compose, AddBackgroundNoise, PitchShift, TimeStretch, Trim

# Abaikan peringatan resample yang mengganggu terminal
warnings.filterwarnings("ignore", message=".*had to be resampled.*")

POSITIVE_DIR = "training_data/positive"
NOISE_DIR    = "ESC-50/audio"
OUTPUT_DIR   = "training_data/positive_augmented"
USER_VOICES_DIR = "training_data/user_voices"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(USER_VOICES_DIR, exist_ok=True)

augment = Compose([
    AddBackgroundNoise(sounds_path=NOISE_DIR, min_snr_db=5, max_snr_db=20, p=0.8),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.4),
    Trim(top_db=30, p=0.3),
])

# ── 1. Augment TTS-generated positive samples ──
tts_files = glob.glob(os.path.join(POSITIVE_DIR, "*.wav"))
print(f"Augmenting {len(tts_files)} TTS-generated files...")
for i, fp in enumerate(tts_files):
    audio, sr = sf.read(fp)
    # Pastikan audio mono agar tidak crash
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    aug = augment(audio.astype(np.float32), sample_rate=sr)
    out = os.path.join(OUTPUT_DIR, f"aug_tts_{i:04d}.wav")
    sf.write(out, aug, sr)
    if i % 100 == 0:
        print(f"  Augmented TTS {i}/{len(tts_files)}")

# ── 2. Augment user's custom voice samples ──
user_files = glob.glob(os.path.join(USER_VOICES_DIR, "*.wav")) + glob.glob(os.path.join(USER_VOICES_DIR, "*.mp3"))
if len(user_files) > 0:
    print(f"Found {len(user_files)} user voice files. Augmenting each 100 times for balance...")
    aug_count = 0
    samples_per_file = 50
    for file_idx, fp in enumerate(user_files):
        audio, sr = sf.read(fp)
        # Pastikan audio mono agar tidak crash
        if len(audio.shape) > 1:
            audio = audio[:, 0]
        for sample_idx in range(samples_per_file):
            aug = augment(audio.astype(np.float32), sample_rate=sr)
            out = os.path.join(OUTPUT_DIR, f"aug_user_{file_idx:02d}_{sample_idx:04d}.wav")
            sf.write(out, aug, sr)
            aug_count += 1
        print(f"  Finished augmenting user file {file_idx+1}/{len(user_files)} (+{samples_per_file} samples)")
    print(f"User voice augmentation complete. Generated {aug_count} human-based samples.")
else:
    print("No user voice files found in training_data/user_voices. Please put your custom human wav files there if you want them augmented.")

print("Augmentation complete.")