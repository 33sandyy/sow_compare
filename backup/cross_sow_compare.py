import os
import json
import faiss
import numpy as np
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from readFiles import extract_number_heading


# ==========================
# CONFIGURATION
# ==========================
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_DIR = os.path.join(STATIC_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_BIENCODER = "sentence-transformers/all-mpnet-base-v2"
MODEL_CROSS = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FAISS_INDEX_PATH = os.path.join(STATIC_DIR, "sow_faiss.index")
METADATA_PATH = os.path.join(STATIC_DIR, "sow_metadata.json")
ALL_HEADINGS_PATH = os.path.join(STATIC_DIR, "sow_all_headings.json")
OUTPUT_JSON_PATH = os.path.join(STATIC_DIR, "sow_comparison_report.json")

THRESHOLD_STRICT = 0.65
THRESHOLD_SOFT = 0.50


# ==========================
# HELPERS
# ==========================
def log_to_file(content, prefix):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{prefix}_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"🧾 Log saved → {path}")


def normalize_heading(h: str):
    """Normalize heading by removing punctuation and lowercasing."""
    h = h.lower()
    h = re.sub(r'[^a-z0-9\s]+', ' ', h)
    h = re.sub(r'\s+', ' ', h)
    return h.strip()


def load_template():
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Template metadata missing — run sow_rag_index.py first.")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


# ==========================
# MAIN COMPARISON
# ==========================
def compare_uploaded_sow(uploaded_docx):
    print(f"🔹 Loading CrossEncoder model: {MODEL_CROSS}")
    cross_encoder = CrossEncoder(MODEL_CROSS)

    print(f"🔹 Loading BiEncoder model: {MODEL_BIENCODER}")
    biencoder = SentenceTransformer(MODEL_BIENCODER)

    template_meta = load_template()
    template_headings_norm = [normalize_heading(t["section_heading"]) for t in template_meta]

    with open(ALL_HEADINGS_PATH, "r", encoding="utf-8") as f:
        all_template_headings = [normalize_heading(h) for h in json.load(f)]

    # Precompute template embeddings (for efficiency)
    template_texts = [
        f"{t['section_heading']}\n{t['chunk_text']}" for t in template_meta
    ]
    template_embs = biencoder.encode(template_texts, convert_to_numpy=True)

    print(f"📘 Reading uploaded SOW: {uploaded_docx}")
    uploaded_sections = extract_number_heading(os.path.join(STATIC_DIR, uploaded_docx))

    # Log uploaded sections
    section_log = ["Uploaded SOW Section Summary", "=" * 80]
    for sec_num, heading, text in uploaded_sections:
        section_log.append(f"[{sec_num}] {heading}\n{text[:250]}...\n{'-'*80}")
    log_to_file("\n".join(section_log), "uploaded_sow_sections")

    comparison_results = []
    comparison_log = []

    # ==========================
    # SECTION-BY-SECTION CHECK
    # ==========================
    for sec_num, heading, content in uploaded_sections:

        h_norm = normalize_heading(heading)

        # 1️⃣ Find matching template headings (substring)
        matched_indices = [
            i for i, th in enumerate(template_headings_norm)
            if h_norm in th or th in h_norm
        ]

        # Case A: Heading exists in template but template chunk is blank
        if h_norm in all_template_headings and len(matched_indices) == 0:
            comparison_results.append({
                "uploaded_section_number": sec_num,
                "uploaded_heading": heading,
                "uploaded_text": content,
                "matched_template_section": heading,
                "matched_template_text": "",
                "similarity_score": 0.0,
                "status": "Template Blank"
            })
            comparison_log.append(
                f"[{sec_num}] {heading} → Template Blank (heading exists but template has no content)"
            )
            continue

        # Case B: Heading does NOT exist anywhere in the template
        if not matched_indices:
            comparison_results.append({
                "uploaded_section_number": sec_num,
                "uploaded_heading": heading,
                "uploaded_text": content,
                "matched_template_section": "",
                "matched_template_text": "",
                "similarity_score": 0.0,
                "status": "Deviated"
            })
            comparison_log.append(
                f"[{sec_num}] {heading} → Deviated (no matching heading found)"
            )
            continue

        # 2️⃣ CrossEncoder comparison only for matched headings
        best_score = -1
        best_match = None

        for idx in matched_indices:
            t = template_meta[idx]
            t_content = t["chunk_text"].strip()

            if not t_content:
                continue  # skip empty template blocks

            query_pair = (f"{heading}\n{content}", f"{t['section_heading']}\n{t_content}")
            score = cross_encoder.predict([query_pair])[0]

            if score > best_score:
                best_score = score
                best_match = t

        # 3️⃣ If all matched template sections were blank
        if best_match is None:
            first_heading = template_meta[matched_indices[0]]["section_heading"]

            comparison_results.append({
                "uploaded_section_number": sec_num,
                "uploaded_heading": heading,
                "uploaded_text": content,
                "matched_template_section": first_heading,
                "matched_template_text": "",
                "similarity_score": 0.0,
                "status": "Template Blank"
            })
            comparison_log.append(
                f"[{sec_num}] {heading} → Template Blank (all matched headings had no template content)"
            )
            continue

        # 4️⃣ Assign classification
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
            "matched_template_section": best_match["section_heading"],
            "matched_template_text": best_match["chunk_text"],
            "similarity_score": float(np.round(best_score, 4)),
            "status": status
        })

        comparison_log.append(
            f"[{sec_num}] {heading} → {status} | Score={best_score:.3f} | Match: {best_match['section_heading']}"
        )

    # Save logs
    log_to_file("\n".join(comparison_log), "sow_compare_log")

    # Save final JSON
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
