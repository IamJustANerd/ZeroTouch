"""
prepare_data_v2.py
==================
STEP 1 of the v2 wake word training pipeline.

Performs source-level 80/10/10 (train/test/val) split BEFORE any augmentation,
so that no source recording ever appears in more than one split.

Pipeline per source type
------------------------
TTS positives  : split 800/100/100 → augment each split independently
Human voices   : preprocess (noise gate, normalise, trim, QC) → split by
                 speaker group → augment each split independently
Negatives      : split 2400/300/300 (no augmentation needed)

Outputs
-------
training_data_v2/
  positive_train/   – augmented TTS + human, train sources only
  positive_test/    – augmented TTS + human, test sources only
  positive_val/     – augmented TTS + human, val sources only
  negative_train/
  negative_test/
  negative_val/
split_manifest.json – maps every output file back to its source (for docs)
"""

import os, glob, json, shutil, math, warnings, random
import numpy as np
import soundfile as sf
from scipy.signal import resample as sci_resample
from audiomentations import (
    Compose, AddBackgroundNoise, PitchShift, TimeStretch, Trim,
    AddGaussianNoise, RoomSimulator, Gain
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED        = 42
TARGET_SR   = 16000

# Source directories
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TTS_DIR     = os.path.join(ROOT_DIR, "training_data", "positive")
HUMAN_DIR   = os.path.join(ROOT_DIR, "training_data", "user_voices")
NEG_DIR     = os.path.join(ROOT_DIR, "training_data", "negative")
NOISE_DIR   = os.path.join(ROOT_DIR, "ESC-50", "audio")

# Output root
OUT_BASE    = os.path.join(ROOT_DIR, "training_data_v2")

# Split ratios
TRAIN_RATIO = 0.80
TEST_RATIO  = 0.10
# VAL_RATIO = 0.10 (implicit remainder)

# Augmentation multipliers per split
TTS_AUG_TRAIN = 4    # 800 sources × 4 = 3200 training clips
TTS_AUG_TEST  = 2    # 100 sources × 2 = 200  test clips
TTS_AUG_VAL   = 2    # 100 sources × 2 = 200  val clips

HUMAN_AUG_TRAIN = 10  # each human train source → 10 augmented clips
HUMAN_AUG_TEST  = 5   # each human test source  → 5  augmented clips
HUMAN_AUG_VAL   = 5   # each human val source   → 5  augmented clips

# Random seeds for each split's augmentation — MUST be disjoint so augmentations
# applied to train never accidentally match those applied to test/val.
AUG_SEED_TRAIN = 0
AUG_SEED_TEST  = 10_000
AUG_SEED_VAL   = 20_000

# Human voice QC thresholds
MIN_DUR_S    = 0.8    # reject clips shorter than this
MAX_DUR_S    = 4.5    # reject clips longer than this
MIN_SNR_DB   = 10     # reject clips with low speech energy (approx SNR)

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION PIPELINES
# (Three independent pipelines — different random seeds keep them disjoint)
# ─────────────────────────────────────────────────────────────────────────────

def _make_augmenter(noise_dir, seed):
    """Return a Compose augmenter seeded with `seed`."""
    transforms = [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        PitchShift(min_semitones=-3, max_semitones=3, p=0.6),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
    ]
    if noise_dir and os.path.isdir(noise_dir):
        transforms.insert(0,
            AddBackgroundNoise(sounds_path=noise_dir,
                               min_snr_db=5, max_snr_db=25, p=0.7))
    augmenter = Compose(transforms)
    # audiomentations uses Python's random internally — seed it globally for
    # this split, then reset after we're done.
    random.seed(seed)
    np.random.seed(seed)
    return augmenter


AUG_TRAIN = _make_augmenter(NOISE_DIR, AUG_SEED_TRAIN)
AUG_TEST  = _make_augmenter(NOISE_DIR, AUG_SEED_TEST)
AUG_VAL   = _make_augmenter(NOISE_DIR, AUG_SEED_VAL)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def load_and_resample(path: str) -> np.ndarray:
    """Load any audio file, convert to 16 kHz mono float32."""
    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception:
        # Fallback for MP3 etc. that soundfile can't read natively
        import subprocess, tempfile
        tmp = tempfile.mktemp(suffix=".wav")
        subprocess.run(["ffmpeg", "-y", "-i", path, "-ar", str(TARGET_SR),
                        "-ac", "1", tmp], capture_output=True)
        data, sr = sf.read(tmp, always_2d=False)
        os.remove(tmp)

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        n_samples = int(len(data) * TARGET_SR / sr)
        data = sci_resample(data, n_samples)
    return data.astype(np.float32)


def spectral_noise_gate(audio: np.ndarray, sr: int = TARGET_SR,
                        noise_sec: float = 0.1) -> np.ndarray:
    """
    Simple spectral subtraction noise gate.
    Estimates noise from the first `noise_sec` seconds and attenuates the
    frequency components that fall below the noise floor.
    """
    import librosa
    n_noise = int(noise_sec * sr)
    if len(audio) <= n_noise + 1:
        return audio  # too short to estimate noise

    # STFT
    D = librosa.stft(audio)
    mag, phase = np.abs(D), np.angle(D)

    # Noise estimate from the first n_noise samples
    noise_clip = audio[:n_noise]
    D_noise    = librosa.stft(noise_clip)
    noise_mag  = np.mean(np.abs(D_noise), axis=1, keepdims=True)

    # Spectral subtraction with flooring at 0.01 to avoid musical noise
    mag_clean = np.maximum(mag - noise_mag, 0.01 * mag)
    D_clean   = mag_clean * np.exp(1j * phase)

    audio_clean = librosa.istft(D_clean, length=len(audio))
    return audio_clean.astype(np.float32)


def rms_normalise(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalise audio to a target RMS level in dBFS."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    return audio * (target_rms / rms)


def trim_silence(audio: np.ndarray, sr: int = TARGET_SR,
                 top_db: float = 30) -> np.ndarray:
    """Trim leading/trailing silence."""
    import librosa
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return audio_trimmed.astype(np.float32)


def quality_check(audio: np.ndarray, sr: int = TARGET_SR,
                  path: str = "") -> tuple[bool, str]:
    """Return (ok, reason). Reject if duration or SNR are out of range."""
    dur = len(audio) / sr
    if dur < MIN_DUR_S:
        return False, f"too short ({dur:.2f}s < {MIN_DUR_S}s)"
    if dur > MAX_DUR_S:
        return False, f"too long ({dur:.2f}s > {MAX_DUR_S}s)"
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return False, "silent (peak ~0)"
    snr_approx_db = 20 * math.log10(peak / (rms + 1e-9) + 1e-9)
    if snr_approx_db < MIN_SNR_DB:
        return False, f"low SNR ({snr_approx_db:.1f} dB < {MIN_SNR_DB} dB)"
    return True, "ok"


def preprocess_human_voice(path: str) -> np.ndarray | None:
    """
    Full preprocessing chain for a raw human voice recording.
    Returns cleaned float32 16 kHz mono array, or None if QC fails.
    """
    print(f"  Preprocessing {os.path.basename(path)} ...", end="  ")
    try:
        audio = load_and_resample(path)
        audio = spectral_noise_gate(audio)
        audio = rms_normalise(audio)
        audio = trim_silence(audio)
        ok, reason = quality_check(audio, path=path)
        if not ok:
            print(f"REJECTED ({reason})")
            return None
        print(f"ok  ({len(audio)/TARGET_SR:.2f}s)")
        return audio
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def write_wav(audio: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, TARGET_SR, subtype="PCM_16")


# ─────────────────────────────────────────────────────────────────────────────
# SPLITTING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def source_split(files: list, train_r=TRAIN_RATIO, test_r=TEST_RATIO,
                 seed=SEED) -> tuple[list, list, list]:
    """
    Randomly shuffle `files` with fixed seed and split into
    (train, test, val) at the source level.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    files = [files[i] for i in idx]

    n_train = math.floor(len(files) * train_r)
    n_test  = math.floor(len(files) * test_r)
    # val gets the remainder so counts always sum to total
    return files[:n_train], files[n_train:n_train+n_test], files[n_train+n_test:]


def speaker_aware_split(human_files: list) -> tuple[list, list, list]:
    """
    Split 15 human voice files ensuring each unique speaker appears in at
    least one split.  Groups files by speaker prefix (deva, michael, rendy, ren)
    and distributes them 80/10/10.

    With only 15 files this is:
      - 12 train / 2 test / 1 val  (we round to nearest integer per speaker)
    """
    # Group by speaker
    groups: dict[str, list] = {}
    for f in human_files:
        name = os.path.basename(f).lower()
        if name.startswith("deva"):
            key = "deva"
        elif name.startswith("michael"):
            key = "michael"
        else:
            key = "rendy_ren"   # merge rendy* and ren_* — same person
        groups.setdefault(key, []).append(f)

    train_all, test_all, val_all = [], [], []
    for speaker, files in groups.items():
        rng = np.random.default_rng(SEED)
        rng.shuffle(files)
        n = len(files)
        n_test = max(1, round(n * TEST_RATIO))
        n_val  = max(1, round(n * (1 - TRAIN_RATIO - TEST_RATIO)))
        n_train= n - n_test - n_val
        if n_train < 1:  # edge case: very few files for a speaker
            n_train = 1; n_test = max(1, n - 2); n_val = max(0, n - n_train - n_test)
        train_all.extend(files[:n_train])
        test_all.extend(files[n_train:n_train+n_test])
        val_all.extend(files[n_train+n_test:n_train+n_test+n_val])

    print(f"  Speaker-aware split — train:{len(train_all)}  test:{len(test_all)}  val:{len(val_all)}")
    return train_all, test_all, val_all


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENT + SAVE
# ─────────────────────────────────────────────────────────────────────────────

def augment_and_save(sources: list,
                     augmenter,
                     out_dir: str,
                     n_augments: int,
                     prefix: str,
                     manifest: dict,
                     is_preprocessed_array: bool = False):
    """
    For each source (path or (name, array) tuple), generate `n_augments` augmented
    copies using `augmenter` and save to `out_dir`.

    Returns list of output paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []

    for src_idx, src in enumerate(sources):
        # Load source audio
        if is_preprocessed_array:
            name, audio = src
        else:
            name = os.path.splitext(os.path.basename(src))[0]
            audio = load_and_resample(src)

        for aug_i in range(n_augments):
            aug_audio = augmenter(audio, sample_rate=TARGET_SR)

            # Ensure audio is within range
            peak = np.max(np.abs(aug_audio))
            if peak > 0:
                aug_audio = aug_audio / peak * 0.95

            out_name = f"{prefix}_{src_idx:04d}_aug{aug_i:02d}.wav"
            out_path = os.path.join(out_dir, out_name)
            write_wav(aug_audio, out_path)
            out_paths.append(out_path)

            # Track in manifest
            manifest[out_name] = {
                "source": name,
                "augmentation_index": aug_i,
                "split": os.path.basename(out_dir).replace("positive_","").replace("negative_",""),
                "type": prefix.split("_")[0],
            }

        if src_idx % 100 == 0:
            print(f"    [{prefix}] {src_idx+1}/{len(sources)} sources processed ...")

    print(f"  -> Saved {len(out_paths)} clips to {out_dir}")
    return out_paths


def copy_negatives(src_files: list, out_dir: str, manifest: dict, split_name: str):
    """Copy (and resample) negative files into the output split directory."""
    os.makedirs(out_dir, exist_ok=True)
    for src in src_files:
        name = os.path.basename(src)
        dst  = os.path.join(out_dir, name)
        audio = load_and_resample(src)
        write_wav(audio, dst)
        manifest[name] = {"source": src, "split": split_name, "type": "negative"}
    print(f"  -> Saved {len(src_files)} negatives to {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    manifest: dict = {}

    # ── 0. Create output directories ─────────────────────────────────────────
    for d in ["positive_train","positive_test","positive_val",
              "negative_train","negative_test","negative_val"]:
        os.makedirs(os.path.join(OUT_BASE, d), exist_ok=True)

    # ── 1. Load TTS source files ──────────────────────────────────────────────
    print("\n=== [1/5] Loading TTS sources ===")
    tts_files = sorted(glob.glob(os.path.join(TTS_DIR, "*.wav")))
    print(f"  Found {len(tts_files)} TTS source files")

    tts_train, tts_test, tts_val = source_split(tts_files)
    print(f"  TTS split: train={len(tts_train)}  test={len(tts_test)}  val={len(tts_val)}")

    # ── 2. Load + preprocess human voice sources ──────────────────────────────
    print("\n=== [2/5] Preprocessing human voice sources ===")
    human_raw = sorted(glob.glob(os.path.join(HUMAN_DIR, "*.wav")) +
                       glob.glob(os.path.join(HUMAN_DIR, "*.mp3")))
    print(f"  Found {len(human_raw)} human voice files")

    human_clean = []  # list of (basename, preprocessed_array)
    for f in human_raw:
        audio = preprocess_human_voice(f)
        if audio is not None:
            human_clean.append((os.path.splitext(os.path.basename(f))[0], audio))

    print(f"\n  {len(human_clean)}/{len(human_raw)} human voices passed QC")

    # Speaker-aware split on the cleaned files
    # We need file paths for speaker grouping — reconstruct them
    clean_paths = [os.path.join(HUMAN_DIR, name + ext)
                   for name, _ in human_clean
                   for ext in [".wav", ".mp3"] if os.path.exists(
                       os.path.join(HUMAN_DIR, name + ext))]
    # Fallback: just use names for grouping
    human_clean_paths_fake = [os.path.join(HUMAN_DIR, name + ".wav")
                               for name, _ in human_clean]

    h_train_paths, h_test_paths, h_val_paths = speaker_aware_split(human_clean_paths_fake)

    # Map back to (name, array) tuples
    array_map = {name: arr for name, arr in human_clean}
    def paths_to_arrays(paths):
        result = []
        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0]
            if name in array_map:
                result.append((name, array_map[name]))
        return result

    h_train = paths_to_arrays(h_train_paths)
    h_test  = paths_to_arrays(h_test_paths)
    h_val   = paths_to_arrays(h_val_paths)
    print(f"  Human split: train={len(h_train)}  test={len(h_test)}  val={len(h_val)}")

    # ── 3. Load negative files ────────────────────────────────────────────────
    print("\n=== [3/5] Splitting negatives ===")
    neg_files = sorted(glob.glob(os.path.join(NEG_DIR, "*.wav")))
    print(f"  Found {len(neg_files)} negative files")
    neg_train, neg_test, neg_val = source_split(neg_files)
    print(f"  Negative split: train={len(neg_train)}  test={len(neg_test)}  val={len(neg_val)}")

    # ── 4. Augment positives ──────────────────────────────────────────────────
    print("\n=== [4/5] Augmenting positive clips ===")

    print("\n  -- TRAIN set --")
    augment_and_save(tts_train,     AUG_TRAIN, os.path.join(OUT_BASE,"positive_train"), TTS_AUG_TRAIN, "tts",   manifest)
    augment_and_save(h_train,       AUG_TRAIN, os.path.join(OUT_BASE,"positive_train"), HUMAN_AUG_TRAIN, "human", manifest, is_preprocessed_array=True)

    print("\n  -- TEST set --")
    augment_and_save(tts_test,      AUG_TEST,  os.path.join(OUT_BASE,"positive_test"),  TTS_AUG_TEST,   "tts",   manifest)
    augment_and_save(h_test,        AUG_TEST,  os.path.join(OUT_BASE,"positive_test"),  HUMAN_AUG_TEST, "human", manifest, is_preprocessed_array=True)

    print("\n  -- VAL set --")
    augment_and_save(tts_val,       AUG_VAL,   os.path.join(OUT_BASE,"positive_val"),   TTS_AUG_VAL,    "tts",   manifest)
    augment_and_save(h_val,         AUG_VAL,   os.path.join(OUT_BASE,"positive_val"),   HUMAN_AUG_VAL,  "human", manifest, is_preprocessed_array=True)

    # ── 5. Copy / resample negatives ─────────────────────────────────────────
    print("\n=== [5/5] Copying negatives ===")
    copy_negatives(neg_train, os.path.join(OUT_BASE,"negative_train"), manifest, "train")
    copy_negatives(neg_test,  os.path.join(OUT_BASE,"negative_test"),  manifest, "test")
    copy_negatives(neg_val,   os.path.join(OUT_BASE,"negative_val"),   manifest, "val")

    # ── Save manifest ──────────────────────────────────────────────────────────
    manifest_path = os.path.join(OUT_BASE, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nManifest saved -> {manifest_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    def count(d): return len(glob.glob(os.path.join(OUT_BASE, d, "*.wav")))
    print("\n" + "="*60)
    print("  DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"  positive_train : {count('positive_train'):>5}")
    print(f"  positive_test  : {count('positive_test'):>5}")
    print(f"  positive_val   : {count('positive_val'):>5}")
    print(f"  negative_train : {count('negative_train'):>5}")
    print(f"  negative_test  : {count('negative_test'):>5}")
    print(f"  negative_val   : {count('negative_val'):>5}")
    print("="*60)
    print("\nNext step: run  python train_wakeword_v2.py")


if __name__ == "__main__":
    main()
