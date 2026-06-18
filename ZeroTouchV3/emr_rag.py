"""
emr_rag.py  –  Offline RAG Engine for ZeroTouch
=================================================
Indexes all patient PDFs at startup using sentence-transformers + ChromaDB.
Exposes a single query() function for semantic search over all patient records.
Everything runs 100% offline.
"""

import os
import threading
import logging

logger = logging.getLogger(__name__)

ROOT        = os.path.dirname(os.path.abspath(__file__))
PATIENT_DIR = os.path.join(ROOT, "data")
INDEX_DIR   = os.path.join(ROOT, "emr_index")

_ready      = False
_collection = None
_embedder   = None
_lock       = threading.Lock()


def _chunk_text(text: str, size: int = 350, overlap: int = 60) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def _build_index():
    global _ready, _collection, _embedder
    try:
        import pypdf
        import chromadb
        from sentence_transformers import SentenceTransformer

        print("[rag] Loading multilingual embedding model…")
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        client = chromadb.PersistentClient(path=INDEX_DIR)
        try:
            client.delete_collection("emr")
        except Exception:
            pass
        _collection = client.create_collection("emr", metadata={"hnsw:space": "cosine"})

        docs, ids, metas = [], [], []
        if not os.path.isdir(PATIENT_DIR):
            print(f"[rag] Patient directory not found: {PATIENT_DIR}")
            return

        for patient_folder in os.listdir(PATIENT_DIR):
            pdf_path = os.path.join(PATIENT_DIR, patient_folder, "Medical_Record.pdf")
            if not os.path.exists(pdf_path):
                continue
            reader   = pypdf.PdfReader(pdf_path)
            raw_text = "\n".join(p.extract_text() or "" for p in reader.pages)
            if not raw_text.strip():
                continue
            patient_name = patient_folder.replace("_", " ")
            for i, chunk in enumerate(_chunk_text(raw_text)):
                docs.append(chunk)
                ids.append(f"{patient_folder}_{i}")
                metas.append({"patient": patient_name, "source": pdf_path})
            print(f"[rag]   Indexed: {patient_name}")

        if docs:
            embeddings = _embedder.encode(docs).tolist()
            _collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)

        with _lock:
            _ready = True
        print(f"[rag] Ready — {len(docs)} chunks across {len(set(m['patient'] for m in metas))} patients.")

    except Exception as e:
        print(f"[rag] Initialization failed: {e}")


def initialize():
    """Call once at app startup. Runs indexing in a daemon thread."""
    t = threading.Thread(target=_build_index, daemon=True, name="rag-indexer")
    t.start()


def is_ready() -> bool:
    with _lock:
        return _ready


def query(question: str, top_k: int = 3) -> str | None:
    """
    Semantic search over all patient records.
    Returns a formatted context string, or None if not ready / nothing relevant found.
    """
    with _lock:
        if not _ready:
            return None
    try:
        q_vec   = _embedder.encode([question]).tolist()
        results = _collection.query(
            query_embeddings=q_vec,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs      = results["documents"][0]
        metas_res = results["metadatas"][0]
        dists     = results["distances"][0]

        # cosine distance < 0.75 → genuinely relevant.
        # 0.65 was too tight: informal/mispronounced queries scored just above it.
        relevant = [
            (doc, meta)
            for doc, meta, dist in zip(docs, metas_res, dists)
            if dist < 0.75
        ]
        if not relevant:
            return None

        parts = [f"[Rekam Medis: {m['patient']}]\n{d}" for d, m in relevant]
        return "\n\n---\n\n".join(parts)

    except Exception as e:
        print(f"[rag] Query error: {e}")
        return None
