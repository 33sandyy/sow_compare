import os
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from readFiles import extract_number_heading

# === CONFIG ===
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"   # better semantic accuracy
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_DIR = os.path.join(STATIC_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(STATIC_DIR, "sow_faiss.index")
METADATA_PATH = os.path.join(STATIC_DIR, "sow_metadata.json")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


# === Helper: Chunk text ===
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip().replace("\n", " ")
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= 30:      # skip very small chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# === Helper: Logging ===
def log_to_file(content, prefix="sow_log"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{prefix}_{timestamp}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Log saved → {log_path}")


# === Build FAISS Index ===
def build_faiss_index_from_docx(docx_filename):
    docx_path = os.path.join(STATIC_DIR, docx_filename)
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"{docx_filename} not found in static folder")

    print(f"📘 Reading Template SOW: {docx_filename}")
    sections = extract_number_heading(docx_path)
    print(f"Extracted {len(sections)} sections")

    # === Log section extraction ===
    log_lines = [f"SOW Section Extraction Log ({docx_filename})\n{'='*80}\n"]
    for num, heading, content in sections:
        snippet = (content[:400] + "...") if len(content) > 400 else content
        log_lines.append(f"[Section {num}] {heading}\n{snippet}\n{'-'*80}\n")
    log_to_file("\n".join(log_lines), prefix="sow_sections")

    # === Load model ===
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # === Prepare section-level embeddings ===
    section_texts = [f"Heading: {h}. Content: {c}" for (_, h, c) in sections]
    print(f"Generating {len(section_texts)} section embeddings...")
    embeddings = model.encode(section_texts, show_progress_bar=True, convert_to_numpy=True)

    metadata = [
        {
            "section_number": num,
            "section_heading": heading,
            "chunk_id": f"{num}_1",
            "chunk_text": content
        }
        for (num, heading, content) in sections
    ]

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ FAISS index saved → {FAISS_INDEX_PATH}")
    print(f"✅ Metadata saved → {METADATA_PATH}")



if __name__ == "__main__":
    sow_file = "SOWTemplateDoc.docx"
    build_faiss_index_from_docx(sow_file)
