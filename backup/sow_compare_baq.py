import os
import json
import faiss
import numpy as np
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer
from readFiles import extract_number_heading


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
ALL_HEADINGS_PATH = os.path.join(STATIC_DIR, "sow_all_headings.json")
OUTPUT_JSON_PATH = os.path.join(STATIC_DIR, "sow_comparison_report.json")

THRESHOLD_STRICT = 0.65
THRESHOLD_SOFT = 0.50


# ==========================
# HELPERS
# ==========================
def log_to_file(content, filename_prefix="sow_compare"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{filename_prefix}_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"🧾 Log saved → {path}")


def normalize_heading(h: str):
    """Remove punctuation, normalize spaces, lowercase."""
    h = h.lower()
    h = re.sub(r'[^a-z0-9\s]+', ' ', h)
    h = re.sub(r'\s+', ' ', h).strip()
    return h


def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Template FAISS index or metadata not found. Run sow_rag_index.py first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def get_embedding_model():
    print(f"🔹 Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def compute_similarity(emb1, emb2):
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return np.dot(emb1, emb2)


# ==========================
# MAIN COMPARISON
# ==========================
def compare_uploaded_sow(uploaded_docx):
    model = get_embedding_model()
    index, template_meta = load_faiss_index()

    # Precompute embeddings
    template_texts = [f"Heading: {t['section_heading']}. Content: {t['chunk_text']}" for t in template_meta]
    template_embs = model.encode(template_texts, convert_to_numpy=True)
    template_headings = [normalize_heading(t["section_heading"]) for t in template_meta]

    if not os.path.exists(ALL_HEADINGS_PATH):
        raise FileNotFoundError("Missing sow_all_headings.json — rebuild index first")

    with open(ALL_HEADINGS_PATH, "r", encoding="utf-8") as f:
        all_template_headings = [normalize_heading(h) for h in json.load(f)]

    print(f"📘 Reading uploaded SOW: {uploaded_docx}")
    uploaded_sections = extract_number_heading(os.path.join(STATIC_DIR, uploaded_docx))

    if not uploaded_sections:
        raise ValueError("No sections extracted from uploaded SOW — check heading styles.")

    # Log summary
    section_log = [f"Uploaded SOW Section Summary: {uploaded_docx}", "=" * 80]
    for n, h, c in uploaded_sections:
        section_log.append(f"[{n}] {h}\n{c[:200]}...\n{'-'*80}")
    log_to_file("\n".join(section_log), filename_prefix="uploaded_sow_sections")

    comparison_results, compare_log = [], []

    for sec_num, heading, content in uploaded_sections:
        if not content.strip():
            continue

        hkey = normalize_heading(heading)

        # --- STEP 1: find matching headings
        matched_indices = [i for i, th in enumerate(template_headings)
                        if hkey in th or th in hkey]

        # --- STEP 2: handle missing chunks but existing heading
        if not matched_indices:
            if hkey in all_template_headings:
                # heading exists but no text in template
                comparison_results.append({
                    "uploaded_section_number": sec_num,
                    "uploaded_heading": heading,
                    "uploaded_text": content,
                    "matched_template_section": heading,
                    "matched_template_text": "",
                    "similarity_score": 0.0,
                    "status": "Template Blank"
                })
                compare_log.append(f"[{sec_num}] {heading} → Template Blank (exists but no text in template)")
            else:
                # completely missing heading
                comparison_results.append({
                    "uploaded_section_number": sec_num,
                    "uploaded_heading": heading,
                    "uploaded_text": content,
                    "matched_template_section": "",
                    "matched_template_text": "",
                    "similarity_score": 0.0,
                    "status": "Deviated"
                })
                compare_log.append(f"[{sec_num}] {heading} → Deviated (no heading match in template)")
            continue

        # --- STEP 3: compare only if we have chunks
        query_text = f"Heading: {heading}. Content: {content}"
        query_emb = model.encode([query_text], convert_to_numpy=True)[0]

        sims = [(compute_similarity(query_emb, template_embs[i]), i) for i in matched_indices]
        best_score, best_idx = max(sims, key=lambda x: x[0]) if sims else (0, None)
        best_match = template_meta[best_idx] if best_idx is not None else None

        # --- STEP 4: assign status
        if best_score >= THRESHOLD_STRICT:
            status = "Matched"
        elif best_score >= THRESHOLD_SOFT:
            status = "Partially Matched"
        else:
            status = "Deviated"

        comparison_results.append({
            "uploaded_section_number": sec_num,
            "uploaded_heading": heading,
            "uploaded_text": content,
            "matched_template_section": best_match["section_heading"] if best_match else "",
            "matched_template_text": best_match["chunk_text"][:500] if best_match else "",
            "similarity_score": float(np.round(best_score, 4)),
            "status": status
        })

        compare_log.append(f"[{sec_num}] {heading} → {status} | Score={best_score:.3f} | Match: {best_match['section_heading'] if best_match else 'None'}")


    # Save results
    log_to_file("\n".join(compare_log), filename_prefix="sow_compare_log")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Comparison completed → {OUTPUT_JSON_PATH}")
    return comparison_results


if __name__ == "__main__":
    uploaded_file = "SOWDev - Copy.docx"
    results = compare_uploaded_sow(uploaded_file)
    print("\n=== Preview ===")
    for r in results[:5]:
        print(f"[{r['uploaded_heading']}] → {r['matched_template_section']} ({r['status']}) {r['similarity_score']}")
