"""
sow_compare_local.py
-----------------------------------
Compares an uploaded SOW document with the template FAISS index
using **only local search per section**.

Classifies each section/chunk as:
 - Matched (>= THRESHOLD_STRICT)
 - Partially Matched (>= THRESHOLD_SOFT)
 - Deviated (< THRESHOLD_SOFT)
"""

import os
import json
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from readFiles import extract_number_heading
from sow_rag_index import chunk_text  # reuse chunking from indexer

# ==========================
# CONFIGURATION
# ==========================
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_DIR = os.path.join(STATIC_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(STATIC_DIR, "sow_faiss.index")
METADATA_PATH = os.path.join(STATIC_DIR, "sow_metadata.json")
OUTPUT_JSON_PATH = os.path.join(STATIC_DIR, "sow_comparison_report.json")

THRESHOLD_STRICT = 0.45  # high confidence match
THRESHOLD_SOFT = 0.35    # weaker but acceptable match


# ==========================
# LOGGING HELPER
# ==========================
def log_to_file(content, filename_prefix="sow_compare"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{filename_prefix}_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Log saved → {path}")


# ==========================
# LOAD TEMPLATE INDEX
# ==========================
def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Template FAISS index or metadata not found. Please run sow_rag_index.py first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


# ==========================
# EMBEDDING HELPER
# ==========================
def get_embedding_model():
    print(f"🔹 Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def compute_similarity(emb1, emb2):
    """Compute cosine similarity between two numpy vectors."""
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1_norm, emb2_norm)


# ==========================
# MAIN COMPARISON LOGIC
# ==========================
def compare_uploaded_sow(uploaded_docx):
    """
    Compares uploaded SOW sections to template index (local only).
    """
    model = get_embedding_model()
    index, template_meta = load_faiss_index()

    print(f"Reading uploaded SOW: {uploaded_docx}")
    uploaded_sections = extract_number_heading(os.path.join(STATIC_DIR, uploaded_docx))

    # === Log Uploaded SOW Section Summary ===
    section_log_lines = []
    section_log_lines.append(f"Uploaded SOW Section Extraction Summary: {uploaded_docx}")
    section_log_lines.append("=" * 100)
    for sec_num, heading, content in uploaded_sections:
        preview = (content[:250] + "...") if len(content) > 250 else content
        section_log_lines.append(f"[{sec_num}] {heading}\n{preview}\n{'-'*100}")
    log_to_file("\n".join(section_log_lines), filename_prefix="uploaded_sow_sections")
    print(f"Uploaded SOW sections logged ({len(uploaded_sections)} sections).")

    if not uploaded_sections:
        raise ValueError("No sections extracted from uploaded SOW. Check document formatting or heading styles.")

    # Prepare template data grouped by section heading
    template_by_heading = {}
    for t in template_meta:
        heading = t["section_heading"].strip().lower()
        template_by_heading.setdefault(heading, []).append(t)

    comparison_results = []
    compare_log = []

    for sec_num, heading, content in uploaded_sections:
        if not content.strip():
            continue

        heading_key = heading.strip().lower()
        local_chunks = template_by_heading.get(heading_key, [])

        # if not local_chunks:
        #     # No corresponding section in template → Deviated
        #     comparison_results.append({
        #         "uploaded_section_number": sec_num,
        #         "uploaded_heading": heading,
        #         "uploaded_chunk_id": f"{sec_num}_1",
        #         "uploaded_text": content,
        #         "matched_template_section": "",
        #         "matched_template_text": "",
        #         "similarity_score": 0.0,
        #         "status": "Deviated"
        #     })
        #     compare_log.append(f"[{sec_num}] {heading} → Deviated (no local section)")
        #     continue
        if not local_chunks:
            # Template has no matching section OR blank section
            # Check if the heading exists in template metadata but has empty text
            template_has_blank = any(
                t["section_heading"].strip().lower() == heading.strip().lower()
                and not t["chunk_text"].strip()
                for t in template_meta
            )

            if template_has_blank:
                status = "Template Blank"
                msg = f"({heading}) exists but no reference content"
            else:
                status = "Deviated"
                msg = "(no matching section found in template)"

            comparison_results.append({
                "uploaded_section_number": sec_num,
                "uploaded_heading": heading,
                "uploaded_chunk_id": f"{sec_num}_1",
                "uploaded_text": content,
                "matched_template_section": heading if template_has_blank else "",
                "matched_template_text": "" if template_has_blank else "",
                "similarity_score": 0.0,
                "status": status
            })

            compare_log.append(f"[{sec_num}] {heading} → {status} {msg}")
            continue


        # === Chunk uploaded content ===
        chunks = chunk_text(content, 500, 100)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{sec_num}_{idx+1}"
            emb = model.encode([chunk], convert_to_numpy=True)

            best_score, best_match = 0, None
            for t in local_chunks:
                t_emb = model.encode([t["chunk_text"]], convert_to_numpy=True)
                score = compute_similarity(emb[0], t_emb[0])
                if score > best_score:
                    best_score = score
                    best_match = t

            # === Label based on thresholds ===
            if best_score >= THRESHOLD_STRICT:
                status = "Matched"
            elif best_score >= THRESHOLD_SOFT:
                status = "Partially Matched"
            else:
                status = "Deviated"

            result = {
                "uploaded_section_number": sec_num,
                "uploaded_heading": heading,
                "uploaded_chunk_id": chunk_id,
                "uploaded_text": chunk,
                "matched_template_section": best_match["section_heading"] if best_match else "",
                "matched_template_text": best_match["chunk_text"][:500] if best_match else "",
                "similarity_score": float(np.round(best_score, 4)),
                "status": status
            }
            comparison_results.append(result)

            compare_log.append(
                f"[{chunk_id}] {heading} → {status} | Score={best_score:.3f} | "
                f"Match: {result['matched_template_section']}"
            )

    # ==========================
    # SAVE OUTPUT
    # ==========================
    log_to_file("\n".join(compare_log), filename_prefix="sow_compare_local_log")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Local-only comparison completed. Report saved → {OUTPUT_JSON_PATH}")
    print(f"Total Chunks Compared: {len(comparison_results)}")

    return comparison_results


# ==========================
# RUN EXAMPLE
# ==========================
if __name__ == "__main__":
    uploaded_file = "SOWDev.docx"
    results = compare_uploaded_sow(uploaded_file)

    print("\n=== Summary Preview ===")
    for r in results[:5]:
        print(f"[{r['uploaded_heading']}] → {r['matched_template_section']}")
        print(f"Similarity: {r['similarity_score']} | {r['status']}\n")
