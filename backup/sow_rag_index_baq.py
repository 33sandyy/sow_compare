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

    all_chunks = []
    chunk_log = []

    for sec_num, heading, content in sections:
        if not content.strip():
            continue
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            entry = {
                "section_number": sec_num,
                "section_heading": heading,
                "chunk_id": f"{sec_num}_{i+1}",
                "chunk_text": chunk
            }
            all_chunks.append(entry)
            
            all_section_headings = [h.strip() for _, h, _ in sections]
            with open(os.path.join(STATIC_DIR, "sow_all_headings.json"), "w", encoding="utf-8") as f:
                json.dump(all_section_headings, f, indent=2, ensure_ascii=False)

            # preview = (chunk[:250] + "...") if len(chunk) > 250 else chunk
            # chunk_log.append(f"[Chunk {entry['chunk_id']}] {heading}\n{preview}\n{'-'*80}\n")
            # Prepare logging text
            preview = (content[:300] + "...") if len(content) > 300 else content
            chunk_log.append(
                f"[Section {sec_num}] {heading}\n"
                f"Chunk ID: {entry['chunk_id']}\n"
                f"Length: {len(content)} chars\n"
                f"Preview:\n{preview}\n{'-'*100}\n"
            )

    log_to_file("Template SOW Chunk Log\n" + "="*100 + "\n" + "\n".join(chunk_log), prefix="template_sow_chunks")

    # log_to_file("SOW Chunk Log\n" + "="*80 + "\n" + "\n".join(chunk_log), prefix="sow_chunks")
    print(f"Total chunks: {len(all_chunks)}")

    # === Embed with heading context ===
    print("Generating embeddings with heading context...")
    texts = [f"Heading: {c['section_heading']}. Content: {c['chunk_text']}" for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # === Save metadata ===
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # === Build & save FAISS ===
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"✅ FAISS index saved → {FAISS_INDEX_PATH}")
    print(f"✅ Metadata saved → {METADATA_PATH}")


if __name__ == "__main__":
    sow_file = "SOW_temp - Copy.docx"
    build_faiss_index_from_docx(sow_file)
