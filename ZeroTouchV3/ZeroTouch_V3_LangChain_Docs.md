# Dokumentasi LangChain Stack - ZeroTouch V3

Berdasarkan analisis file di direktori `ZeroTouchV3`, berikut adalah dokumentasi lengkap dari *stack* teknologi dan implementasi LangChain yang digunakan pada sistem V3.

## 🛠️ LangChain Stack di ZeroTouch V3

Sistem V3 sudah tidak menggunakan rantai (chains) lawas, melainkan menggunakan arsitektur berbasis *Agent* dengan **LangGraph** dan model lokal **Ollama**.

**Stack Utama:**
1. **Core:** `langchain-core` (untuk manajemen pesan dan tools)
2. **LLM Provider:** `langchain-ollama` (`ChatOllama`)
3. **Agent Framework:** `langgraph` (menggunakan `create_react_agent` untuk eksekusi logika ReAct secara dinamis)
4. **Local LLM Model:** Ollama menjalankan `llama3.2:latest` (berjalan pada port `127.0.0.1:11434` dengan temperature 0 untuk determinisme).
5. **RAG Backend:** (Offline/Custom) Menggunakan `chromadb` sebagai *vector database* dan `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) untuk pemrosesan semantik dari ekstraksi teks PDF (`pypdf`).

---

## 📚 Dokumentasi Tools (Fungsi Agent)

AI Agent di `langchain_agent.py` dilengkapi dengan 8 buah *tools* terintegrasi yang dapat dieksekusi secara otonom melalui sistem *callbacks*.

| Nama Tool | Parameter | Deskripsi & Fungsi |
| :--- | :--- | :--- |
| `buka_aplikasi` | `nama_aplikasi` | Membuka aplikasi sistem secara langsung (misal: word, whatsapp, notepad, chrome, explorer). |
| `buka_file` | `nama_pasien`, `jenis_file` | Mencari dan menampilkan file/dokumen spesifik ('scan', 'record', atau 'any') milik seorang pasien ke layar. |
| `buka_foto_pasien` | `nama_pasien` | Shortcut khusus untuk mencari dan langsung menampilkan file gambar/scan/x-ray milik pasien tertentu. |
| `buka_info_pasien` | `pertanyaan` | Memicu pencarian RAG (`emr_rag.py`). Akan mencari jawaban secara semantik di *vector database* (ChromaDB) dari *knowledge base* rekam medis pasien. |
| `zoom_layar` | `arah`, `region`, `jumlah` | Melakukan Zoom-in/Zoom-out pada layar berdasarkan arah ("in"/"out"), area yang difokuskan (seperti "center", "top-left", "bot-right"), dan jumlah tingkatan zoom (1-5). |
| `buat_dan_tulis_file` | `nama_file`, `konten` | Memungkinkan AI untuk membuat file teks baru dan menuliskan konten di dalamnya secara langsung ke *filesystem*. |
| `mulai_notulensi` | - | Mengaktifkan fitur rekam notulensi latar belakang. Semua perkataan dokter pasca perintah ini akan dicatat diam-diam. |
| `berhenti_notulensi` | - | Menghentikan rekam notulensi, merapikan formatnya, lalu menyimpan catatannya secara otomatis ke penyimpanan lokal. |

---

## ⚙️ Konfigurasi Behavior & Memory

**1. Memory State (Session History)**
- Agent memelihara konteks percakapan di dalam list `session_history` (memuat objek `HumanMessage` dan `AIMessage`). Ini memungkinkan LLM tetap mengingat konteks pertanyaan atau identitas pasien sebelumnya saat berinteraksi (memori jangka pendek).

**2. System Prompt Protection**
- Agent menggunakan proteksi `SystemMessage` yang ketat agar llama 3.2 tidak berhalusinasi mengembalikan format *raw JSON* (suatu isu umum ketika memanggil *tools* pada model Llama kecil).
- Persona dipatok sebagai **"Jarvis, asisten bedah AI cerdas"** dengan panduan untuk membalas dokter menggunakan satu kalimat pendek, langsung ke inti (contoh: *"Baik, gambar telah diperbesar"*), dan menanyakan klarifikasi alih-alih membuat asumsi sendiri jika perintah dirasa kurang jelas.
