# ZeroTouch: Non-Contact Interface for Contamination-Free Operating Rooms

ZeroTouch is an intelligent, offline-first interface designed for surgical environments. It allows doctors to interact with Electronic Medical Records (EMR) and operate digital medical imaging tools without breaking the sterile field, thereby minimizing the risk of cross-contamination (fomites). 

The system relies on a combination of **Hand Gesture Control**, **Voice Activation**, and **Local LLM-powered RAG (Retrieval-Augmented Generation)** to fetch patient records dynamically.

---

## 🏗️ System Architecture

The project is divided into several modular components that are orchestrated by a central unified launcher:

### 1. Unified Launcher (`zerotouch_v0.py`)
The heart of the application. It provides a transparent, non-intrusive PyQt6 UI overlay that displays the camera feed, speech-to-text transcriptions, and Jarvis's chat logs.
- **State Management:** Manages `WAKE` and `SLEEP` states.
- **Threading:** Runs the STT and Gesture engines in separate background threads to ensure the UI and tracking remain ultra-responsive.
- **LLM/TTS Pipeline:** Routes transcribed voice commands to the RAG engine and then to the local Llama 3.2 model for natural language synthesis.

### 2. Hand Gesture Engine (`ZeroTouch/hand_tracker.py`)
Responsible for reading camera frames, identifying hand landmarks using Google's **MediaPipe**, and classifying the gesture using a custom **TensorFlow Lite (TFLite)** model.
- **Natural Mapping:**
  - `Open Palm`: Move the cursor (smooth tracking via Exponential Moving Average).
  - `Pointing`: Left-click.
  - `Fist`: Hold and drag (uses raw OS `ctypes` for ultra-low latency).
  - `Pinch`: Scroll / Zoom in medical viewers.
- **Clutch Mechanism:** Prevents accidental triggers. The user must hold an `Open Palm` for 2 seconds to transition the system from SLEEP to ACTIVE.

### 3. Voice & STT Engine 
Provides ultra-fast, local speech-to-text capabilities.
- **Wake Word Detection:** Uses `openwakeword` to constantly listen for "Hey Jarvis" with minimal CPU overhead.
- **Transcription:** Uses `faster-whisper` to accurately transcribe Indonesian speech.

### 4. Offline EMR RAG Engine (`emr_rag.py`)
A highly secure, offline semantic search engine for patient records. 
- **Indexing:** Reads all PDFs in the `OpenClawProject/Patient/` directory using `pypdf`.
- **Embedding:** Uses `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) to convert medical text into mathematical vectors.
- **Database:** Stores the vectors in **ChromaDB** (`emr_index/`). 
- **Retrieval:** When a doctor asks a question, this module fetches the most relevant paragraphs from the PDFs to provide context for the LLM.

### 5. App Bridge (`host_bridge.py`)
A secure Flask-based API that allows the ZeroTouch AI to launch specific, whitelisted Windows host applications (e.g., Word, Notepad, WhatsApp) or open specific medical imaging files securely.

---

## 🚀 How to Run

### Prerequisites
1. Ensure **Ollama** is installed and running locally with the `llama3.2:latest` model pulled.
   ```bash
   ollama run llama3.2:latest
   ```
2. Install the required Python packages:
   ```bash
   pip install PyQt6 opencv-python mediapipe tensorflow pyautogui faster-whisper openwakeword sounddevice numpy requests pypdf chromadb sentence-transformers
   ```

### Execution
The `zerotouch_v0.py` launcher will automatically spin up `host_bridge.py` and the RAG indexing engine in the background.

```bash
python zerotouch_v0.py
```

---

## 🗣️ Interacting with the System

### Waking the System
By default, the system starts in `SLEEP` mode (Red Border) to prevent accidental clicks during surgery. You can wake it via two methods:
1. **Gesture:** Hold an `Open Palm` up to the camera for 2 seconds.
2. **Voice:** Say *"Hey Jarvis"*.
*(The border will turn Green, indicating it is actively listening for commands and tracking the cursor).*

### Putting the System to Sleep
1. **Gesture:** Hold a `Fist` for 2 seconds.
2. **Voice:** Say *"Sleep"*, *"Tidur"*, or *"Berhenti"*.

### Asking about Patient Records (RAG)
When in the `WAKE` state, simply ask a question in natural language.
- *"Jarvis, apa diagnosis utama Budi Santoso?"*
- *"Apakah Agus punya riwayat alergi obat?"*

*Behind the scenes, Jarvis searches the ChromaDB index, finds the relevant PDF chunks, and uses Llama 3.2 to read the records and respond via text-to-speech.*

---

## 🔒 Security & Privacy (HIPAA / Permenkes Compliance)

This system is designed specifically with Edge Computing principles for medical environments:
- **No Cloud Services:** Whisper, openwakeword, Llama 3.2, and ChromaDB all run 100% locally on the terminal's hardware.
- **No Image Retention:** Camera frames used for hand tracking are instantly overwritten in RAM. No photos or video feeds are ever saved to the disk.
- **Air-Gapped Ready:** The system does not require an active internet connection to transcribe speech, process gestures, or answer EMR queries.
