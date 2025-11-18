# sow_compare.py
import os
import json
import faiss
import numpy as np
import re
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
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

# baseline thresholds for refined sentence-level score
THRESHOLD_STRICT = 0.55
THRESHOLD_SOFT = 0.45

# number of FAISS candidates to evaluate with sentence-level refinement
TOP_K_CANDIDATES = 3

# how many top sentence matches to average when computing refined score
TOP_SENT_MATCHES = 3


# ==========================
# HELPERS
# ==========================
def log_to_file(content, prefix="sow_compare"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{prefix}_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"🧾 Log saved → {path}")

def split_sentences(text):
    text = text.strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def normalize_heading(h: str):
    h = h.lower()
    h = re.sub(r'[^a-z0-9\s]+', ' ', h)
    h = re.sub(r'\s+', ' ', h).strip()
    return h

def load_index_and_metadata():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("FAISS index or metadata missing. Run sow_tag_index.py first.")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# ==========================
# SCORING helpers
# ==========================
def sentence_level_refinement(model, uploaded_text, template_text):
    """
    Compute sentence-level similarity using SBERT cosine similarity.
    """
    uploaded_sents = split_sentences(uploaded_text)
    template_sents = split_sentences(template_text)

    if not uploaded_sents or not template_sents:
        return 0.0, []

    # embed sentences
    u_embs = model.encode(uploaded_sents, convert_to_numpy=True)
    t_embs = model.encode(template_sents, convert_to_numpy=True)

    # SBERT util.cos_sim returns a PyTorch tensor → convert to numpy
    sims = util.cos_sim(u_embs, t_embs).cpu().numpy()  # shape (U, T)

    # flatten and pick top matches
    flat = sims.flatten()
    topk = min(TOP_SENT_MATCHES, len(flat))
    top_indices = np.argpartition(-flat, topk - 1)[:topk]
    top_scores = [float(flat[i]) for i in top_indices]

    refined_score = float(np.mean(top_scores)) if top_scores else 0.0
    refined_score = max(0.0, refined_score)  # clip negatives

    # prepare explainability pairs
    matched_pairs = []
    for idx in top_indices:
        u_idx = idx // sims.shape[1]
        t_idx = idx % sims.shape[1]
        matched_pairs.append({
            "uploaded_sentence": uploaded_sents[u_idx],
            "template_sentence": template_sents[t_idx],
            "score": float(sims[u_idx, t_idx])
        })

    matched_pairs = sorted(matched_pairs, key=lambda x: -x["score"])
    return refined_score, matched_pairs


# ==========================
# MAIN COMPARISON
# ==========================
def compare_uploaded_sow(uploaded_docx):
    # load models and index/metadata
    print(f"Loading SBERT model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    index, metadata = load_index_and_metadata()
    # precompute template section texts array for quick access
    template_texts = [f"{m['section_heading']}\n{m['chunk_text']}" for m in metadata]
    template_headings_norm = [normalize_heading(m['section_heading']) for m in metadata]

    # read uploaded doc
    uploaded_path = os.path.join(STATIC_DIR, uploaded_docx)
    print(f"Reading uploaded SOW: {uploaded_docx}")
    uploaded_sections = extract_number_heading(uploaded_path)
    if not uploaded_sections:
        raise ValueError("No sections extracted from uploaded SOW. Check heading styles.")

    # log extracted uploaded headings
    section_log = [f"Uploaded SOW Section Summary: {uploaded_docx}", "="*80]
    for n, h, c in uploaded_sections:
        section_log.append(f"[{n}] {h}\n{c[:250]}...\n{'-'*80}")
    log_to_file("\n".join(section_log), prefix="uploaded_sow_sections")

    results = []
    compare_logs = []

    # ==========================
    # Process each uploaded section
    # ==========================
    for sec_num, heading, content in uploaded_sections:
        heading_norm = normalize_heading(heading)

        # 1) find candidate template headings by normalized substring
        candidate_indices = [i for i, th in enumerate(template_headings_norm) if heading_norm in th or th in heading_norm]

        # if no candidate headings at all, check all headings file (maybe heading exists but no content)
        if not candidate_indices:
            # check if heading exists in template(all headings)
            all_headings_path = os.path.join(STATIC_DIR, "sow_all_headings.json")
            if os.path.exists(all_headings_path):
                with open(all_headings_path, "r", encoding="utf-8") as f:
                    all_headings = [normalize_heading(x) for x in json.load(f)]
                if heading_norm in all_headings:
                    # heading exists but likely template had empty chunk -> Template Blank
                    results.append({
                        "uploaded_section_number": sec_num,
                        "uploaded_heading": heading,
                        "uploaded_text": content,
                        "matched_template_section": heading,
                        "matched_template_text": "",
                        "faiss_score": 0.0,
                        "refined_score": 0.0,
                        "status": "Template Blank",
                        "matched_sentences": []
                    })
                    compare_logs.append(f"[{sec_num}] {heading} → Template Blank (heading present but no template text)")
                    continue
            # completely missing heading
            results.append({
                "uploaded_section_number": sec_num,
                "uploaded_heading": heading,
                "uploaded_text": content,
                "matched_template_section": "",
                "matched_template_text": "",
                "faiss_score": 0.0,
                "refined_score": 0.0,
                "status": "Deviated",
                "matched_sentences": []
            })
            compare_logs.append(f"[{sec_num}] {heading} → Deviated (no heading match)")
            continue

        # 2) Compute embedding for the uploaded section (single vector)
        query_text = f"{heading}\n{content}"
        query_emb = model.encode([query_text], convert_to_numpy=True)
        # normalize for inner-product
        qnorm = np.linalg.norm(query_emb, axis=1, keepdims=True)
        qnorm[qnorm == 0] = 1e-9
        query_emb = query_emb / qnorm

        # 3) FAISS search among all vectors, but we'll prefer restricting to candidate_indices
        # fast approach: search full index top_k then filter by candidate_indices, fallback to candidate brute-force if needed
        top_k = max(TOP_K_CANDIDATES, len(candidate_indices))
        D, I = index.search(query_emb.astype("float32"), top_k)  # D: inner product distances
        D = D[0].tolist()
        I = I[0].tolist()

        # Build candidate list intersection preserving order by FAISS score
        candidate_list = []
        for idx, score in zip(I, D):
            if idx in candidate_indices:
                candidate_list.append((idx, float(score)))
            if len(candidate_list) >= TOP_K_CANDIDATES:
                break

        # If FAISS didn't return any of the candidate_indices (rare), fallback to brute-force scoring on candidate_indices
        if not candidate_list:
            for idx in candidate_indices:
                # compute inner product with precomputed template embeddings if available:
                # we don't have precomputed embeddings in this function, so do model encode for template_texts[idx]
                emb = model.encode([template_texts[idx]], convert_to_numpy=True)
                emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
                score = float(np.dot(query_emb[0], emb[0]))
                candidate_list.append((idx, score))
            # sort descending
            candidate_list = sorted(candidate_list, key=lambda x: -x[1])[:TOP_K_CANDIDATES]

        # 4) For each candidate, perform sentence-level refinement and pick the best refined score
        best_refined = -1.0
        best_idx = None
        best_faiss = 0.0
        best_matched_sentences = []

        for idx, faiss_score in candidate_list:
            t_text = template_texts[idx]
            refined_score, matched_pairs = sentence_level_refinement(model, content, metadata[idx]['chunk_text'])
            # adjust refined score to be within [-1,1] (cos_sim returns in [-1,1])
            # If similarities are negative (rare), clip to 0
            refined_score = max(0.0, refined_score)

            # choose best by refined_score primarily, if tie use faiss_score
            if (refined_score > best_refined) or (abs(refined_score - best_refined) < 1e-6 and faiss_score > best_faiss):
                best_refined = refined_score
                best_idx = idx
                best_faiss = faiss_score
                best_matched_sentences = matched_pairs

        if best_idx is None:
            # fallback
            results.append({
                "uploaded_section_number": sec_num,
                "uploaded_heading": heading,
                "uploaded_text": content,
                "matched_template_section": "",
                "matched_template_text": "",
                "faiss_score": 0.0,
                "refined_score": 0.0,
                "status": "Deviated",
                "matched_sentences": []
            })
            compare_logs.append(f"[{sec_num}] {heading} → Deviated (no candidate refined match)")
            continue

        # 5) Dynamic thresholds based on template sentence count (short templates => relax threshold)
        tpl_sent_count = len(split_sentences(metadata[best_idx]['chunk_text']))
        strict = THRESHOLD_STRICT
        soft = THRESHOLD_SOFT
        if tpl_sent_count <= 2:
            strict -= 0.10  # relax
            soft -= 0.10

        # final classification based on refined_score
        if best_refined >= strict:
            status = "Matched"
        elif best_refined >= soft:
            status = "Partially Matched"
        else:
            status = "Deviated"

        result = {
            "uploaded_section_number": sec_num,
            "uploaded_heading": heading,
            "uploaded_text": content,
            "matched_template_section": metadata[best_idx]['section_heading'],
            "matched_template_text": metadata[best_idx]['chunk_text'],
            "faiss_score": float(np.round(best_faiss, 4)),
            "refined_score": float(np.round(best_refined, 4)),
            "status": status,
            "matched_sentences": best_matched_sentences
        }
        results.append(result)
        compare_logs.append(f"[{sec_num}] {heading} → {status} | refined={best_refined:.3f} | faiss={best_faiss:.3f} | Match: {metadata[best_idx]['section_heading']}")

    # Save logs and JSON
    log_to_file("\n".join(compare_logs), prefix="sow_compare_sentence_refine")
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Section-wise SBERT comparison completed → {OUTPUT_JSON_PATH}")
    return results

# ==========================
# RUN EXAMPLE
# ==========================
if __name__ == "__main__":
    uploaded_file = "SOWDev - Copy.docx"  # change as required
    compare_uploaded_sow(uploaded_file)
