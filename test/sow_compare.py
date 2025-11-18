import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from readFiles import extract_number_heading
from sow_rag_index import chunk_text, METADATA_PATH, FAISS_INDEX_PATH, MODEL_NAME, STATIC_DIR

# === CONFIG ===
THRESHOLD_STRICT = 0.85
THRESHOLD_SOFT = 0.70
OUTPUT_JSON = os.path.join(STATIC_DIR, "sow_comparison_report.json")


def compare_uploaded_sow(uploaded_filename, top_k=1):
    # Load uploaded docx
    docx_path = os.path.join(STATIC_DIR, uploaded_filename)
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"{uploaded_filename} not found in static folder")

    print(f"🔍 Comparing uploaded SOW: {uploaded_filename}")
    uploaded_sections = extract_number_heading(docx_path)
    print(f"Found {len(uploaded_sections)} sections in uploaded SOW")

    # Load template FAISS index and metadata
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError("Template FAISS index or metadata not found. Please build the index first.")

    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        template_metadata = json.load(f)

    comparison_results = []

    for sec_num, heading, content in uploaded_sections:
        if not content.strip():
            continue
        chunks = chunk_text(content)

        for idx, chunk in enumerate(chunks):
            # Embed uploaded chunk
            query_emb = model.encode([chunk], convert_to_numpy=True)
            D, I = index.search(np.array(query_emb).astype("float32"), top_k)

            matched_template = None
            similarity = 0.0

            # Compare with top-1 template chunk
            if len(I[0]) > 0:
                temp_idx = I[0][0]
                if 0 <= temp_idx < len(template_metadata):
                    matched_template = template_metadata[temp_idx]
                    temp_text = matched_template["chunk_text"]
                    template_emb = model.encode([temp_text], convert_to_numpy=True)
                    similarity = util.cos_sim(query_emb, template_emb)[0][0].item()

            # Determine status
            if similarity >= THRESHOLD_STRICT:
                status = "Aligned ✅"
            elif similarity >= THRESHOLD_SOFT:
                status = "Partially aligned ⚠️"
            else:
                status = "Deviated ❌"

            comparison_results.append({
                "uploaded_section_number": sec_num,
                "uploaded_heading": heading,
                "uploaded_chunk_id": f"{sec_num}_{idx+1}",
                "uploaded_text": chunk,
                "matched_template_section": matched_template["section_heading"] if matched_template else "N/A",
                "matched_template_text": matched_template["chunk_text"] if matched_template else "N/A",
                "similarity_score": round(similarity, 4),
                "status": status
            })

    # Save output report
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Comparison complete. Results saved to: {OUTPUT_JSON}")
    return comparison_results


if __name__ == "__main__":
    uploaded_file = "SOWDev.docx"  # <-- your uploaded file name
    results = compare_uploaded_sow(uploaded_file)

    print("\n=== Comparison Summary ===")
    for r in results[:5]:  # show first few
        print(f"\n[Uploaded: {r['uploaded_heading']}] → [Template: {r['matched_template_section']}]")
        print(f"Similarity: {r['similarity_score']} | Status: {r['status']}")
