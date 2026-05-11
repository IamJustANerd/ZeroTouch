import os
import subprocess

def synthesize_indonesian(text: str, output_path: str):
    """
    Synthesize Indonesian text to speech using Piper CLI.
    """
    # Paths to the model and Piper executable
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "id_ID-news_tts-medium.onnx")
    piper_exe = os.path.join(base_dir, "venv", "Scripts", "piper.exe")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    if not os.path.exists(piper_exe):
        raise FileNotFoundError(f"Piper executable not found at: {piper_exe}")

    print(f"Synthesizing: \"{text}\"")

    # Run piper via subprocess
    # We pipe the text to standard input of the piper executable
    try:
        process = subprocess.run(
            [piper_exe, "--model", model_path, "--output_file", output_path],
            input=text.encode("utf-8"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"✅ Audio successfully saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print("❌ Error during synthesis!")
        print(e.stderr.decode("utf-8"))

if __name__ == "__main__":
    # Test text in Indonesian
    test_text = "Selamat siang, saya adalah asisten virtual ProTel. Bagaimana saya bisa membantu Anda hari ini?"
    out_file = "tts-audio/asisten_protel.wav"
    
    synthesize_indonesian(test_text, out_file)
