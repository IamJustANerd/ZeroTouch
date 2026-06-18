import os
import sys
import subprocess
import re

# Fix Windows terminal encoding for emojis
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.utils.medical_dict import MEDICAL_PHONETIC_DICT

def normalize_english_to_indo(text: str) -> str:
    """Menerjemahkan ejaan Inggris ke cara baca Indonesia berdasarkan kamus fonetik medis"""
    for eng_word, indo_phonetic in MEDICAL_PHONETIC_DICT.items():
        text = re.sub(eng_word, indo_phonetic, text, flags=re.IGNORECASE)
    return text

def synthesize_indonesian(text: str, output_path: str, length_scale: float = 1.0, noise_scale: float = 0.667, noise_w: float = 0.8):
    """
    Synthesize Indonesian text to speech using Piper.
    Runs piper as a Python module (sys.executable -m piper) so it works
    in any virtual environment without needing a hardcoded executable path.
    """
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "id_ID-news_tts-medium.onnx")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # Normalisasi teks Inggris ke pelafalan Indonesia
    phonetic_text = normalize_english_to_indo(text)

    print(f"Original Text: \"{text}\"")
    print(f"Synthesizing (Phonetic): \"{phonetic_text}\"")

    try:
        subprocess.run(
            [
                sys.executable, "-m", "piper",
                "--model", model_path,
                "--output_file", output_path,
                "--length_scale", str(length_scale),
                "--noise_scale", str(noise_scale),
                "--noise_w", str(noise_w),
            ],
            input=phonetic_text.encode("utf-8"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"✅ Audio successfully saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print("❌ Error during synthesis!")
        print(e.stderr.decode("utf-8", errors="replace"))

if __name__ == "__main__":
    # Test text in Indonesian mixed with English
    test_text = "Hello, Zero Touch"
    out_file = "tts-audio/hello_zerotouch2.wav"

    # You can tweak these parameters to make the voice sound more natural
    # length_scale: Adjusts speaking speed (lower is faster, e.g., 0.8 for faster, 1.2 for slower) try 0.85-0.95
    # noise_scale: Adjusts voice variability/emotion (default 0.667, try 0.5 to 0.8) try >=0.8
    # noise_w: Adjusts phoneme length variability (default 0.8, try 0.6 to 1.0) try 0.9
    synthesize_indonesian(test_text, out_file, length_scale=0.9, noise_scale=0.8, noise_w=0.9)
