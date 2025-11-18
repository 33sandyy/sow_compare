# sow_tag_index.py
import os
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from readFiles import extract_number_heading
import re

# === CONFIG ===
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"   # SBERT model
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_DIR = os.path.join(STATIC_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(STATIC_DIR, "sow_faiss.index")
METADATA_PATH = os.path.join(STATIC_DIR, "sow_metadata.json")
ALL_HEADINGS_PATH = os.path.join(STATIC_DIR, "sow_all_headings.json")

# === Helpers ===
def log_to_file(content, prefix="sow_log"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{prefix}_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Log saved → {path}")

def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    # basic sentence splitter (works reasonably for English)
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

# === Build FAISS Index ===
def build_faiss_index_from_docx(docx_filename):
    docx_path = os.path.join(STATIC_DIR, docx_filename)
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"{docx_filename} not found in static folder")

    print(f"📘 Reading Template SOW: {docx_filename}")
    sections = extract_number_heading(docx_path)
    print(f"Extracted {len(sections)} sections")

    # Save all headings (even if content empty)
    all_section_headings = [h.strip() for _, h, _ in sections]
    with open(ALL_HEADINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_section_headings, f, indent=2, ensure_ascii=False)

    # Load model
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    metadata = []
    texts_for_embedding = []

    chunk_log = []
    for num, heading, content in sections:
        heading = heading.strip()
        content = content.strip()
        # store the full section as a single chunk (one chunk per section)
        meta = {
            "section_number": num,
            "section_heading": heading,
            "chunk_id": f"{num}_1",
            "chunk_text": content,
            "sentences": split_sentences(content)
        }
        metadata.append(meta)
        texts_for_embedding.append(f"Heading: {heading}\nContent: {content if content else ''}")

        preview = (content[:300] + "...") if len(content) > 300 else content
        chunk_log.append(
            f"[Section {num}] {heading}\nChunk ID: {meta['chunk_id']}\nLength: {len(content)} chars\nPreview:\n{preview}\n{'-'*100}\n"
        )

    log_to_file("Template SOW Section Log\n" + "="*120 + "\n" + "\n".join(chunk_log), prefix="template_sow_sections")
    print(f"Total sections to embed: {len(texts_for_embedding)}")

    # Generate embeddings (bi-encoder)
    print("Generating section embeddings (SBERT)...")
    embeddings = model.encode(texts_for_embedding, show_progress_bar=True, convert_to_numpy=True)

    # Normalize embeddings for cosine-sim via inner product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    embeddings = embeddings / norms

    # Save metadata JSON (no embeddings inside)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Build FAISS index (IndexFlatIP for cosine similarity on normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"✅ FAISS index saved → {FAISS_INDEX_PATH}")
    print(f"✅ Metadata saved → {METADATA_PATH}")

if __name__ == "__main__":
    sow_file = "SOW_temp - Copy.docx"   # change to your template filename
    build_faiss_index_from_docx(sow_file)
