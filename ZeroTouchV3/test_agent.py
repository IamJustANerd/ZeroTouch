import time
import sys
import os

os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_agent import ZeroTouchAgent

def mock_execute_launcher(app_name, path=None):
    print(f"  [MOCK] execute_launcher -> app_name: {app_name}, path: {path}")

def mock_execute_screen(action_dict):
    print(f"  [MOCK] execute_screen -> action: {action_dict}")

def mock_find_patient_file(nama_pasien, jenis_file):
    print(f"  [MOCK] find_patient_file -> nama_pasien: {nama_pasien}, jenis_file: {jenis_file}")
    return f"C:\\mock_path\\{nama_pasien}_{jenis_file}.pdf"

def mock_rag_query(pertanyaan):
    print(f"  [MOCK] rag_query -> pertanyaan: {pertanyaan}")
    return "Mock RAG Answer: Pasien memiliki riwayat hipertensi."

def mock_rag_ready():
    return True

def mock_start_notul():
    print(f"  [MOCK] start_notul -> Notulensi dimulai")

def mock_stop_notul():
    print(f"  [MOCK] stop_notul -> Notulensi dihentikan")

def run_test():
    print("Initializing ZeroTouchAgent with Mock Callbacks...")
    agent = ZeroTouchAgent(
        execute_launcher_cb=mock_execute_launcher,
        execute_screen_cb=mock_execute_screen,
        find_patient_file_cb=mock_find_patient_file,
        rag_query_cb=mock_rag_query,
        rag_ready_cb=mock_rag_ready,
        start_notul_cb=mock_start_notul,
        stop_notul_cb=mock_stop_notul
    )
    print("Agent initialized successfully!\n")

    test_prompts = [
        "Tolong buka aplikasi word",
        "Buka rekam medis pasien bernama Agus Setiawan",
        "Tolong perbesar layarnya di bagian tengah",
        "Mulai notulensi ya",
        "Berhenti merekam notulensi"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print("="*60)
        print(f"Test {i}")
        print(f"Input Prompt : \"{prompt}\"")
        print("-" * 60)
        
        start_time = time.time()
        try:
            response = agent.process_prompt(prompt)
            print("-" * 60)
            print(f"Agent Response: {response}")
        except Exception as e:
            print(f"Error processing prompt: {e}")
        
        print(f"Elapsed Time  : {time.time() - start_time:.2f} detik")
        print("="*60 + "\n")

if __name__ == "__main__":
    run_test()
