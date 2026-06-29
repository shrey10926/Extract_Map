# aws sso login --profile shrey_bedrock
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json, yaml, faiss, numpy as np, pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer

# Reuse data/model/bedrock + normalization primitives already built for suppliers
from supplier_mapping import (
    BASE_DIR, FAISS_DIR, METADATA_DIR, BEDROCK_CONFIG,
    _get_bedrock_client, load_master_dataframe, load_embedding_model,
    lowercase_text, trim_extra_spaces, unicode_normalization,
    safe_separator_normalization, map_supplier_name_from_invoice,
)

# =========================================================
# CONFIG  (same weighting scheme as supplier matching)
# =========================================================
ITEM_WEIGHTS = {"exact": 0.50, "fuzzy": 0.30, "semantic": 0.20}
TOP_K_ITEM_CANDIDATES = 5          # candidate parts kept per line item
# Gating: only a TRUE exact name match auto-accepts; every non-exact item goes to the LLM.

# Messages returned when nothing matches in the DB
SUPPLIER_NOT_FOUND_MSG = "Supplier Name match not found in Database"
ITEM_NOT_FOUND_MSG = "Item Name match not found in Database"

ITEM_RERANK_PROMPT = BASE_DIR / "Prompt" / "item_rerank.yaml"
with open(ITEM_RERANK_PROMPT, encoding="utf-8") as f:
    ITEM_RERANK_SYSTEM_PROMPT = yaml.safe_load(f)["system_prompt"]

ITEM_RERANK_SCHEMA_FILE = BASE_DIR / "Response_Schema" / "item_rerank_schema.json"
with open(ITEM_RERANK_SCHEMA_FILE, "r", encoding="utf-8") as f:
    ITEM_RERANK_RESPONSE_SCHEMA = json.load(f)


# =========================================================
# ITEM NORMALIZATION  (must mirror export_parquet.py exactly)
# =========================================================
def normalize_item_name_fuzzy(text: str) -> str:
    text = lowercase_text(text)
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    text = safe_separator_normalization(text)
    return trim_extra_spaces(text)


def normalize_item_name_semantic(text: str) -> str:
    text = lowercase_text(text)          # NOTE: items lowercase (suppliers do not)
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    return trim_extra_spaces(text)


# =========================================================
# ITEM VECTOR STORE — reconstruct precomputed embeddings (no re-encode)
# =========================================================
_ITEM_RESOURCE_CACHE: Dict[int, Tuple[pd.DataFrame, np.ndarray]] = {}


def load_item_resources(seid: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load the item index + metadata for a site and reconstruct all vectors once.

    `metadata_df` includes Supplier_Id and Part_id; `vectors` is row-aligned to metadata_df.
    """
    if seid in _ITEM_RESOURCE_CACHE:
        return _ITEM_RESOURCE_CACHE[seid]

    faiss_path = FAISS_DIR / f"faiss_seid_{seid}_item_name.index"
    metadata_path = METADATA_DIR / f"metadata_seid_{seid}_item_name.parquet"
    if not faiss_path.exists():
        raise FileNotFoundError(f"Item FAISS index not found: {faiss_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Item metadata parquet not found: {metadata_path}")

    index = faiss.read_index(str(faiss_path))
    metadata_df = pd.read_parquet(metadata_path)

    # Pull already-computed vectors straight out of the flat index (memory copy, no model).
    vectors = index.reconstruct_n(0, index.ntotal).astype(np.float32)

    _ITEM_RESOURCE_CACHE[seid] = (metadata_df, vectors)
    return _ITEM_RESOURCE_CACHE[seid]


# =========================================================
# CANDIDATE BUILDING
# =========================================================
def get_supplier_parts(df_master: pd.DataFrame, seid: int, supplier_id: Any) -> pd.DataFrame:
    """All catalog parts for one supplier at one site (deduped by Part_id)."""
    df = df_master[(df_master["seid"] == seid) &
                   (df_master["Supplier_Id"] == supplier_id)].copy()
    df = df[df["PartName_Descriptive"].fillna("").astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["Part_id"]).reset_index(drop=True)
    df["item_name_fuzzy"] = df["item_name_fuzzy"].fillna("").astype(str)
    df["item_name_semantic"] = df["item_name_semantic"].fillna("").astype(str)
    return df


def get_candidate_vectors(parts_df: pd.DataFrame, metadata_df: pd.DataFrame,
                          vectors: np.ndarray, supplier_id,
                          model: SentenceTransformer) -> np.ndarray:
    """Attach each candidate part's precomputed vector by Part_id, using only this supplier's
    rows in the item index. Encode only the rare miss."""
    # Restrict the index rows to the matched supplier, then map Part_id -> vector row.
    mask = (metadata_df["Supplier_Id"] == supplier_id).to_numpy()
    sup_rows = np.nonzero(mask)[0]
    sup_part_ids = metadata_df.loc[mask, "Part_id"].tolist()
    part_to_row: Dict[Any, int] = {}
    for pid, row in zip(sup_part_ids, sup_rows.tolist()):
        part_to_row.setdefault(_to_int(pid), int(row))   # first wins; dup rows share an identical vector

    out: List[Optional[np.ndarray]] = [None] * len(parts_df)
    miss_pos, miss_text = [], []
    part_ids = parts_df["Part_id"].tolist()
    sem_texts = parts_df["item_name_semantic"].astype(str).tolist()
    for i, pid in enumerate(part_ids):
        row = part_to_row.get(_to_int(pid))
        if row is None:
            miss_pos.append(i); miss_text.append(sem_texts[i].strip())
        else:
            out[i] = vectors[row]
    if miss_text:
        enc = model.encode(miss_text, task="text-matching",
                           convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        for j, i in enumerate(miss_pos):
            out[i] = enc[j]
    return np.vstack(out).astype(np.float32)


# =========================================================
# SCORING  (name only: exact + fuzzy + semantic)
# =========================================================
def score_items_against_parts(query_fuzzy: List[str], query_vecs: np.ndarray,
                              parts_df: pd.DataFrame, cand_vecs: np.ndarray,
                              weights: Optional[Dict[str, float]] = None):
    """Return (final, exact, fuzzy, cosine) score matrices, shape (items x candidates)."""
    weights = weights or ITEM_WEIGHTS
    cand_fuzzy = parts_df["item_name_fuzzy"].astype(str).tolist()

    fuzzy_m = np.asarray(process.cdist(query_fuzzy, cand_fuzzy, scorer=fuzz.WRatio),
                         dtype=np.float32)                                   # 0-100
    q = np.array(query_fuzzy, dtype=object)[:, None]
    c = np.array(cand_fuzzy, dtype=object)[None, :]
    exact_m = (q == c).astype(np.float32)                                   # 0/1
    cosine_m = (query_vecs @ cand_vecs.T).astype(np.float32)               # 0-1 (normalized)

    final = (weights["exact"] * exact_m
             + weights["fuzzy"] * (fuzzy_m / 100.0)
             + weights["semantic"] * cosine_m)
    return final, exact_m, fuzzy_m, cosine_m


def build_item_candidates(parts_df, final, exact_m, fuzzy_m, cosine_m, top_k):
    cands_per_item = []
    for i in range(final.shape[0]):
        order = np.argsort(-final[i])[:top_k]
        rows = []
        for rank, j in enumerate(order, start=1):
            part = parts_df.iloc[int(j)]
            rows.append({
                "rank": rank,
                "Part_id": part.get("Part_id"),
                "part_name": str(part.get("PartName_Descriptive", "")),
                "exact": float(exact_m[i, j]),
                "fuzzy": float(fuzzy_m[i, j]),
                "cosine": float(cosine_m[i, j]),
                "final": float(final[i, j]),
            })
        cands_per_item.append(rows)
    return cands_per_item


# =========================================================
# BATCHED LLM RERANK (one call for all uncertain items, name-only)
# =========================================================
def rerank_items_with_llm(uncertain: List[Tuple[int, list]], line_items: list) -> list:
    blocks = []
    for item_index, cands in uncertain:
        it = line_items[item_index]
        cand_lines = "\n".join(
            f"    Rank {c['rank']}: Part_id={c['Part_id']}, Name=\"{c['part_name']}\", "
            f"Final={c['final']:.4f}, Fuzzy={c['fuzzy']:.1f}, Cosine={c['cosine']:.4f}"
            for c in cands
        ) or "    - (no candidates)"
        blocks.append(
            f"item_index {item_index}: \"{it.get('item_name', 'N/A')}\"\n"
            f"  Candidate parts from this supplier:\n{cand_lines}"
        )

    user_message = (
        "Match each invoice line item to the best candidate catalog part from the "
        "supplier, based only on the item name.\n\n"
        + "\n\n".join(blocks) + "\n\nReturn one decision per item_index."
    )

    client = _get_bedrock_client()
    payload = {
        "modelId": BEDROCK_CONFIG["api"]["model_name"],
        "system": [{"text": ITEM_RERANK_SYSTEM_PROMPT}],
        "messages": [{"role": "user", "content": [{"text": user_message}]}],
        "outputConfig": {"textFormat": {"type": "json_schema", "structure": {
            "jsonSchema": {"name": "item_reranking",
                           "schema": json.dumps(ITEM_RERANK_RESPONSE_SCHEMA)}}}},
        "inferenceConfig": {"temperature": BEDROCK_CONFIG["api"]["temperature"],
                            "maxTokens": BEDROCK_CONFIG["api"]["max_tokens"]},
    }
    response = client.converse(**payload)
    text = (response["output"]["message"]["content"][0]["text"]
            .strip().removeprefix("```json").removeprefix("```").removesuffix("```"))
    return json.loads(text).get("items", [])


# =========================================================
# OUTPUT SHAPING  (response_structure.json item shape)
# =========================================================
def _clean(v):
    """Entities not present in the invoice are null: normalize missing/empty markers to None."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().upper() in ("", "NA", "N/A", "NULL", "NONE"):
        return None
    return v


def _to_int(v):
    """Coerce DB ids (numpy / pandas Int64) to plain int, or None."""
    try:
        if v is None or pd.isna(v):
            return None
        return int(v)
    except Exception:
        return None


def _base(it):
    return {"extracted_item_name": _clean(it.get("item_name")),
            "qty": _clean(it.get("quantity")),
            "unit_price": _clean(it.get("rate")),
            "line_total_price": _clean(it.get("amount"))}


def _matched_item(it, chosen, confidence, reason):
    return {**_base(it),
            "part_id": _to_int(chosen["Part_id"]) if chosen else None,
            "part_name": chosen["part_name"] if chosen else None,
            "match_confidence": confidence,
            "match_reason": reason}


def _not_found_item(it):
    """Item name had no acceptable match in the supplier's catalog."""
    return {**_base(it), "part_id": None, "part_name": None,
            "match_confidence": 0, "match_reason": ITEM_NOT_FOUND_MSG}


def _accepted_item(it, best):
    reason = "Exact name match to catalog part (auto-accepted; no LLM rerank needed)."
    return _matched_item(it, best, 100, reason)


def _resolve_uncertain(it, cands, decision):
    if decision is None:
        return _fallback_item(it, cands)
    if decision.get("no_match") or decision.get("best_candidate_rank") in (None, 0):
        return _not_found_item(it)
    rank = decision.get("best_candidate_rank")
    chosen = next((c for c in cands if c["rank"] == rank), None)
    if chosen is None:
        return _not_found_item(it)
    conf = {"high": 90, "medium": 70, "low": 50}.get(decision.get("confidence", "low"), 50)
    return _matched_item(it, chosen, conf, decision.get("reason", ""))


def _fallback_item(it, cands):
    """LLM unavailable: fall back to the top name-scored candidate."""
    best = cands[0] if cands else None
    if not best:
        return _not_found_item(it)
    reason = f"LLM unavailable; top name-scored candidate (score {best['final']:.2f})."
    return _matched_item(it, best, round(best["final"] * 100), reason)


def _unmatched_all(line_items):
    return [_not_found_item(it) for it in line_items]


# =========================================================
# MAIN — map all line items to the matched supplier's parts
# =========================================================
def map_line_items_from_invoice(invoice_json, supplier_result, df_master, model,
                                top_k_candidates=TOP_K_ITEM_CANDIDATES) -> Dict[str, Any]:
    line_items = invoice_json.get("line_items", []) or []
    seid = int(invoice_json.get("Site_Id"))
    supplier_id = supplier_result.get("best_supplier_id")

    if not line_items:
        return {"supplier_id": supplier_id, "items": []}
    if supplier_id is None or supplier_result.get("no_match"):
        return {"supplier_id": supplier_id, "items": _unmatched_all(line_items)}

    parts_df = get_supplier_parts(df_master, seid, supplier_id)
    if parts_df.empty:
        return {"supplier_id": supplier_id, "items": _unmatched_all(line_items)}

    # Reuse precomputed item vectors (filtered to this supplier); encode the extracted names ONCE.
    metadata_df, vectors = load_item_resources(seid)
    cand_vecs = get_candidate_vectors(parts_df, metadata_df, vectors, supplier_id, model)

    extracted = [str(it.get("item_name") or "") for it in line_items]
    query_fuzzy = [normalize_item_name_fuzzy(t) for t in extracted]
    query_semantic = [normalize_item_name_semantic(t) for t in extracted]
    query_vecs = model.encode(query_semantic, task="text-matching",
                              convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    final, exact_m, fuzzy_m, cosine_m = score_items_against_parts(
        query_fuzzy, query_vecs, parts_df, cand_vecs)
    cands_per_item = build_item_candidates(
        parts_df, final, exact_m, fuzzy_m, cosine_m, top_k_candidates)

    results: List[Optional[dict]] = [None] * len(line_items)
    uncertain = []
    for i, cands in enumerate(cands_per_item):
        best = cands[0] if cands else None
        # Only a TRUE exact name match auto-accepts; everything else goes to the LLM.
        if best and best["exact"] >= 1.0:
            results[i] = _accepted_item(line_items[i], best)
        else:
            uncertain.append((i, cands))

    if uncertain:
        try:
            decisions = rerank_items_with_llm(uncertain, line_items)
            dmap = {d.get("item_index"): d for d in decisions}
            for i, cands in uncertain:
                results[i] = _resolve_uncertain(line_items[i], cands, dmap.get(i))
        except Exception as e:
            print(f"Item LLM rerank failed, using top scored candidate: {e}")
            for i, cands in uncertain:
                results[i] = _fallback_item(line_items[i], cands)

    return {"supplier_id": supplier_id,
            "supplier_name": supplier_result.get("best_supplier_name"),
            "items": results}


# =========================================================
# FINAL RESPONSE ASSEMBLY
# =========================================================
def _supplier_confidence(supplier_result) -> int:
    """0-100 confidence for the supplier match."""
    if supplier_result.get("llm_reranked") and supplier_result.get("llm_reranked_candidates"):
        conf = supplier_result["llm_reranked_candidates"][0].get("confidence", "low")
        return {"high": 90, "medium": 70, "low": 50}.get(conf, 50)
    return round(float(supplier_result.get("best_score", 0.0)) * 100)


def _supplier_reason(supplier_result) -> str:
    if supplier_result.get("llm_reranked") and supplier_result.get("llm_reranked_candidates"):
        return supplier_result["llm_reranked_candidates"][0].get("reason", "")
    q = supplier_result.get("query", {})
    return (f"Matched extracted vendor '{q.get('extracted_supplier', '')}' to "
            f"'{supplier_result.get('best_supplier_name', '')}' "
            f"(ID: {_to_int(supplier_result.get('best_supplier_id'))}) "
            f"with a weighted name score of {float(supplier_result.get('best_score', 0.0)):.2f}.")


def build_invoice_response(invoice_json, df_master, model,
                           invoice_num: int = 1, pages: str = "Page 1") -> Dict[str, Any]:
    """Map one extracted invoice (supplier + items) into the final invoice object."""
    line_items = invoice_json.get("line_items", []) or []
    invoice_obj = {
        "invoice_num": invoice_num,
        "pages": pages,
        "vendor_name": _clean(invoice_json.get("supplier_name")),
        "supplier_id": None,
        "supplier_name": None,
        "supplier_confidence": 0,
        "supplier_match_reason": "",
        "status": "success",
        "error": None,
        "items": [],
    }

    # --- Supplier mapping ---
    supplier_result = map_supplier_name_from_invoice(invoice_json, df_master, model)
    supplier_id = supplier_result.get("best_supplier_id")

    # Supplier not found -> skip item matching, but still echo the line items as not found.
    if supplier_id is None or supplier_result.get("no_match"):
        invoice_obj["supplier_match_reason"] = SUPPLIER_NOT_FOUND_MSG
        invoice_obj["items"] = _unmatched_all(line_items)
        return invoice_obj

    invoice_obj["supplier_id"] = _to_int(supplier_id)
    invoice_obj["supplier_name"] = supplier_result.get("best_supplier_name")
    invoice_obj["supplier_confidence"] = _supplier_confidence(supplier_result)
    invoice_obj["supplier_match_reason"] = _supplier_reason(supplier_result)

    # --- Item mapping (scoped to the matched supplier) ---
    item_result = map_line_items_from_invoice(invoice_json, supplier_result, df_master, model)
    invoice_obj["items"] = item_result["items"]
    return invoice_obj


def build_file_response(invoice_json, df_master, model,
                        file_name: str, total_pages: int = 1) -> Dict[str, Any]:
    """Top-level response for one processed file (wraps a single invoice for now)."""
    try:
        invoice_obj = build_invoice_response(invoice_json, df_master, model)
        return {"file": file_name, "status": "success", "total_pages": total_pages,
                "error": None, "invoices": [invoice_obj]}
    except Exception as e:
        return {"file": file_name, "status": "error", "total_pages": total_pages,
                "error": str(e), "invoices": []}


# # =========================================================
# # EXAMPLE USAGE
# # =========================================================
# if __name__ == "__main__":
#     invoice_json = {
#         "Site_Id": 11,
#         "supplier_name": "Republic Services",
#         "invoice_date": "February 28, 2028",
#         "invoice_number": "0918-006708380",
#         "line_items": [
#             {"item_name": "Waste Container 40 Yd, On Call Service",
#              "quantity": "1.0000", "rate": "174.08", "amount": "174.08"},
#         ],
#         "total_invoice_amount": "174.08",
#     }
#     df_master = load_master_dataframe()
#     model = load_embedding_model()

#     response = build_file_response(
#         invoice_json, df_master, model,
#         file_name="0918-006708380.pdf", total_pages=1
#     )

#     print(json.dumps(response, indent=2, default=str))
