import os
import threading
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class ZeroTouchAgent:
    def __init__(self, execute_launcher_cb, execute_screen_cb, find_patient_file_cb, rag_query_cb, rag_ready_cb, start_notul_cb, stop_notul_cb):
        self.execute_launcher = execute_launcher_cb
        self.execute_screen = execute_screen_cb
        self.find_patient_file = find_patient_file_cb
        self.rag_query = rag_query_cb
        self.rag_ready = rag_ready_cb
        self.start_notul = start_notul_cb
        self.stop_notul = stop_notul_cb
        self.session_history = []
        
        # ==========================================
        # TOOLS DEFINITION
        # ==========================================
        @tool
        def buka_aplikasi(nama_aplikasi: str) -> str:
            """Gunakan ini HANYA jika dokter secara eksplisit meminta untuk membuka aplikasi komputer (contoh: word, whatsapp, excel, notepad, chrome, powerpoint)."""
            print(f"\n[LangChain Tool] 🛠️ Executing: buka_aplikasi(nama_aplikasi={nama_aplikasi!r})")
            app_map = {
                "word": "word", "winword": "word",
                "whatsapp": "whatsapp", "watsap": "whatsapp",
                "notepad": "notepad", "catatan": "notepad",
                "excel": "excel", "powerpoint": "powerpoint",
                "chrome": "chrome", "browser": "chrome",
                "explorer": "explorer", "file manager": "explorer"
            }
            nama_aplikasi = nama_aplikasi.lower()
            for k, v in app_map.items():
                if k in nama_aplikasi:
                    threading.Thread(target=self.execute_launcher, args=(v, None), daemon=True).start()
                    return f"Aplikasi {v} berhasil dibuka."
            return f"Aplikasi {nama_aplikasi} tidak dikenali di sistem."

        @tool
        def buka_file(nama_pasien: str, jenis_file: str = "any") -> str:
            """Mencari dan langsung MEMBUKA file/dokumen milik pasien tertentu ke layar. Argumen 'jenis_file' bisa 'scan', 'record', atau 'any'."""
            print(f"\n[LangChain Tool] 📂 Executing: buka_file(nama_pasien={nama_pasien!r}, jenis_file={jenis_file!r})")
            real_path = self.find_patient_file(nama_pasien, jenis_file)
            if real_path:
                threading.Thread(target=self.execute_launcher, args=("file", real_path), daemon=True).start()
                return f"File {jenis_file} pasien {nama_pasien} berhasil dibuka di layar."
            return f"Tidak ditemukan file jenis '{jenis_file}' untuk pasien {nama_pasien}."

        @tool
        def buka_info_pasien(pertanyaan: str) -> str:
            """Mencari informasi spesifik tentang kondisi, penyakit, atau data rekam medis pasien dari database RAG."""
            print(f"\n[LangChain Tool] 🧠 Executing: buka_info_pasien(pertanyaan={pertanyaan!r})")
            if not self.rag_ready():
                return "Sistem rekam medis sedang dimuat, harap tunggu."
            hasil = self.rag_query(pertanyaan)
            if hasil:
                return f"Hasil pencarian data medis:\n{hasil}"
            return "Informasi tidak ditemukan di dalam rekam medis."

        @tool
        def buat_dan_tulis_file(nama_file: str, konten: str) -> str:
            """Membuat file teks baru dan menyimpannya di sistem komputer."""
            print(f"\n[LangChain Tool] ✍️ Executing: buat_dan_tulis_file(nama_file={nama_file!r})")
            try:
                with open(nama_file, "w", encoding="utf-8") as f:
                    f.write(konten)
                return f"Berhasil membuat file {nama_file}."
            except Exception as e:
                return f"Gagal membuat file: {e}"

        @tool
        def buka_foto_pasien(nama_pasien: str) -> str:
            """Mencari dan langsung membuka file foto, gambar, scan, atau x-ray milik pasien tertentu ke layar."""
            print(f"\n[LangChain Tool] 🖼️ Executing: buka_foto_pasien(nama_pasien={nama_pasien!r})")
            real_path = self.find_patient_file(nama_pasien, "scan")
            if real_path:
                threading.Thread(target=self.execute_launcher, args=("file", real_path), daemon=True).start()
                return f"Foto/Scan pasien {nama_pasien} berhasil dibuka."
            return f"Foto untuk pasien {nama_pasien} tidak ditemukan."

        @tool
        def zoom_layar(arah: str, region: str = "center", jumlah: int = 1) -> str:
            """
            Melakukan Zoom In atau Zoom Out pada layar (contoh: "tolong perbesar layarnya", "zoom out sedikit").
            Argumen 'arah' HARUS berisi "in" atau "out".
            Argumen 'region' adalah posisi zoom (pilih salah satu: 'top-left', 'top-center', 'top-right', 'mid-left', 'center', 'mid-right', 'bot-left', 'bot-center', 'bot-right').
            Argumen 'jumlah' adalah seberapa banyak zoom dilakukan (default: 1, maksimal 5).
            """
            print(f"\n[LangChain Tool] 🔎 Executing: zoom_layar(arah={arah!r}, region={region!r}, jumlah={jumlah!r})")
            
            action = "zoom-in" if arah.lower() == "in" else "zoom-out"
            
            # Panggil callback layar (mengirim request ke host bridge)
            threading.Thread(
                target=self.execute_screen, 
                args=({"action": action, "region": region, "times": jumlah},), 
                daemon=True
            ).start()
            
            return f"Layar di bagian {region} berhasil di-zoom {arah} sebanyak {jumlah} tingkat."

        @tool
        def mulai_notulensi() -> str:
            """
            Mulai fitur perekaman notulensi. Semua perkataan dokter mulai saat ini akan dicatat diam-diam di latar belakang.
            Gunakan tool ini jika dokter meminta 'mulai notulensi', 'tolong rekam', atau 'catat perkataan saya'.
            """
            print(f"\n[LangChain Tool] 📝 Executing: mulai_notulensi()")
            self.start_notul()
            return "Fitur notulensi telah diaktifkan. Semua ucapan dokter sekarang sedang direkam."

        @tool
        def berhenti_notulensi() -> str:
            """
            Berhentikan fitur notulensi. Catatan yang terekam akan dirapikan secara otomatis oleh sistem dan disimpan.
            Gunakan tool ini jika dokter meminta 'berhenti notul', 'stop rekam', atau 'selesai catat'.
            """
            print(f"\n[LangChain Tool] 🛑 Executing: berhenti_notulensi()")
            self.stop_notul()
            return "Fitur notulensi dihentikan. Catatan sedang dirapikan dan disimpan."

        tools = [buka_aplikasi, buka_file, buka_info_pasien, buat_dan_tulis_file, buka_foto_pasien, zoom_layar, mulai_notulensi, berhenti_notulensi]
        
        # PENTING: Gunakan 127.0.0.1 untuk mencegah WinError 10049
        llm = ChatOllama(model="llama3.1:latest", temperature=0, base_url="http://127.0.0.1:11434")
        
        self.agent_executor = create_react_agent(llm, tools)
        
        # System prompt yang ketat agar Llama 3.1 tidak berhalusinasi menghasilkan raw JSON
        self.system_prompt = SystemMessage(content=(
            "Anda adalah Jarvis, asisten bedah AI cerdas. Anda merespons perintah dokter melalui speaker suara. "
            "Tugas Anda: Panggil fungsi (tool) yang tepat secara internal, lalu berikan 1 KALIMAT PENDEK konfirmasi ke dokter. "
            "PENTING: DILARANG KERAS mengucapkan kata-kata seperti 'Note:', 'Catatan:', 'Aturan', atau menjelaskan cara kerja tool. "
            "Jika Anda tidak yakin terhadap suatu perintah atau konteksnya tidak jelas, Anda WAJIB bertanya kembali kepada dokter untuk klarifikasi. JANGAN PERNAH membuat asumsi apa pun. "
            "Langsung ucapkan inti jawaban secara natural. Contoh yang benar: 'Baik, gambar telah diperbesar.' atau 'Maaf, data pasien siapa yang ingin Anda buka?' "
            "Jangan pernah menampilkan format JSON."
        ))

    def process_prompt(self, prompt: str) -> str:
        """Memasukkan prompt user ke LangGraph agent dan mengembalikan response teks akhir."""
        self.session_history.append(HumanMessage(content=prompt))
        
        result = self.agent_executor.invoke({
            "messages": [self.system_prompt] + self.session_history
        })
        
        output = result["messages"][-1].content
        self.session_history.append(AIMessage(content=output))
        
        return output
