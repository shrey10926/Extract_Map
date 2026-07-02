from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import json, yaml, faiss, numpy as np, pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer
from datetime import datetime

from text_normalization import (
    normalize_supplier_name_fuzzy,
    normalize_supplier_name_semantic,
)
from bedrock_utils import get_bedrock_client, parse_model_json, build_system_blocks

DEBUG_DIR = Path("debug_logs")
DEBUG_DIR.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MASTER_PARQUET_PATH = BASE_DIR / "data" / "master_dataset.parquet"

VECTOR_STORE_DIR = Path("vector_store")
FAISS_DIR = VECTOR_STORE_DIR / "faiss_indexes"
METADATA_DIR = VECTOR_STORE_DIR / "metadata"

LOCAL_MODEL_DIR = Path("./models/jina-embeddings-v5-text-small-text-matching")

# Candidate generation config
TOP_K_FUZZY = 10
TOP_K_SEMANTIC = 10
FUZZY_CUTOFF = 80  # minimum WRatio score to keep a fuzzy candidate

# Final weighted scoring config
WEIGHTS = {
    "exact": 0.50,
    "fuzzy": 0.30,
    "semantic": 0.20,
}

# If you want a final top-k output after ranking
TOP_K_FINAL = 10

# LLM Reranking config
# With WEIGHTS (exact 0.5 / fuzzy 0.3 / semantic 0.2) an exact match scores 1.0 while any
# non-exact match caps at 0.5. So any threshold in (0.5, 1.0) auto-accepts only exact matches
# and sends everything else to the LLM reranker. 0.90 keeps a clear margin.
LLM_RERANK_THRESHOLD = 0.90
TOP_K_RERANK = 5
PARTS_PER_CANDIDATE = 3

# High-confidence auto-accept: skip the (~20s) LLM rerank for a NON-exact top candidate
# whose fuzzy AND semantic scores are both very high. Requiring both signals (AND, not OR)
# is deliberate — WRatio can score 100 on a subset/token match (e.g. "ACME" vs "ACME CORP
# HOLDINGS"), so a high cosine is required to confirm the semantics agree too. Tune up to
# send more borderline matches to the LLM, down to skip more of them.
SUPPLIER_AUTO_ACCEPT_FUZZY = 95        # RapidFuzz WRatio (0-100)
SUPPLIER_AUTO_ACCEPT_COSINE = 0.90     # cosine similarity (0-1)
SUPPLIER_AUTO_ACCEPT_CONFIDENCE = 90   # reported supplier_confidence for such matches

with open(BASE_DIR / "bedrock_config_3.yaml", "r") as f:
    BEDROCK_CONFIG = yaml.safe_load(f)

_bedrock_client_cache = None


def _get_bedrock_client():
    global _bedrock_client_cache
    if _bedrock_client_cache is None:
        _bedrock_client_cache = get_bedrock_client(BEDROCK_CONFIG)
    return _bedrock_client_cache


# Normalization primitives + pipelines now live in text_normalization.py
# (single source of truth shared with export_parquet.py and item_mapping.py).


# =========================================================
# MODEL LOADING
# =========================================================

def get_model_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_embedding_model() -> SentenceTransformer:
    if not LOCAL_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Local model directory not found: {LOCAL_MODEL_DIR}. "
            f"Download the model first."
        )

    device = get_model_device()
    model = SentenceTransformer(
        str(LOCAL_MODEL_DIR),
        device=device,
        trust_remote_code=True
    )
    return model


# =========================================================
# DATA LOADING
# =========================================================

def load_master_dataframe() -> pd.DataFrame:
    if not MASTER_PARQUET_PATH.exists():
        raise FileNotFoundError(f"Master parquet not found: {MASTER_PARQUET_PATH}")
    df = pd.read_parquet(MASTER_PARQUET_PATH)
    return df


# =========================================================
# VECTOR STORE LOADING
# =========================================================

_SUPPLIER_RESOURCE_CACHE: Dict[int, Tuple[faiss.Index, pd.DataFrame]] = {}


def load_supplier_resources(seid: int) -> Tuple[faiss.Index, pd.DataFrame]:
    """
    Loads:
      - faiss_seid_<seid>_supplier_name.index
      - metadata_seid_<seid>_supplier_name.parquet
    """
    if seid in _SUPPLIER_RESOURCE_CACHE:
        return _SUPPLIER_RESOURCE_CACHE[seid]

    faiss_path = FAISS_DIR / f"faiss_seid_{seid}_supplier_name.index"
    metadata_path = METADATA_DIR / f"metadata_seid_{seid}_supplier_name.parquet"

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata parquet not found: {metadata_path}")

    index = faiss.read_index(str(faiss_path))
    metadata_df = pd.read_parquet(metadata_path)

    _SUPPLIER_RESOURCE_CACHE[seid] = (index, metadata_df)
    return index, metadata_df


# =========================================================
# CANDIDATE GENERATION
# =========================================================

def prepare_supplier_site_df(df_master: pd.DataFrame, seid: int) -> pd.DataFrame:
    """
    Filter by seid and keep only rows useful for supplier mapping.
    """
    required_cols = ["seid", "Supplier_Id", "Supplier_Name", "supplier_name_fuzzy", "supplier_name_semantic"]
    missing = [c for c in required_cols if c not in df_master.columns]
    if missing:
        raise KeyError(f"Missing required columns in master dataframe: {missing}")

    df_site = df_master[df_master["seid"] == seid].copy()

    # Keep only rows where Supplier_Id and Supplier_Name exist
    df_site = df_site[df_site["Supplier_Id"].notna()].copy()
    df_site["Supplier_Name"] = df_site["Supplier_Name"].fillna("").astype(str).str.strip()
    df_site = df_site[df_site["Supplier_Name"] != ""].copy()

    # Normalize / clean helper columns
    df_site["supplier_name_fuzzy"] = df_site["supplier_name_fuzzy"].fillna("").astype(str).str.strip()
    df_site["supplier_name_semantic"] = df_site["supplier_name_semantic"].fillna("").astype(str).str.strip()

    return df_site.reset_index(drop=True)


def get_exact_supplier_candidates(
    df_site: pd.DataFrame,
    query_fuzzy: str
) -> pd.DataFrame:
    """
    Exact match on supplier_name_fuzzy.
    """
    if not query_fuzzy:
        return pd.DataFrame(columns=[
            "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score", "matched_via"
        ])

    exact_rows = df_site[df_site["supplier_name_fuzzy"] == query_fuzzy].copy()
    if exact_rows.empty:
        return pd.DataFrame(columns=[
            "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score", "matched_via"
        ])

    exact_rows = exact_rows[["seid", "Supplier_Id", "Supplier_Name"]].drop_duplicates().copy()
    exact_rows["exact_match"] = 1.0
    exact_rows["fuzzy_score"] = 100.0
    exact_rows["cosine_score"] = 1.0
    exact_rows["matched_via"] = "exact"

    return exact_rows


def get_fuzzy_supplier_candidates(
    df_site: pd.DataFrame,
    query_fuzzy: str,
    top_k: int = TOP_K_FUZZY,
    cutoff: int = FUZZY_CUTOFF
) -> pd.DataFrame:
    """
    WRatio against supplier_name_fuzzy.

    Uses unique fuzzy strings for scoring, then expands back to all rows that share
    the same fuzzy-normalized supplier string.
    """
    if not query_fuzzy:
        return pd.DataFrame(columns=[
            "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score", "matched_via"
        ])

    choice_df = (
        df_site[["supplier_name_fuzzy"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if choice_df.empty:
        return pd.DataFrame(columns=[
            "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score", "matched_via"
        ])

    choices = choice_df["supplier_name_fuzzy"].tolist()

    matches = process.extract(
        query_fuzzy,
        choices,
        scorer=fuzz.WRatio,
        limit=top_k,
        score_cutoff=cutoff
    )

    if not matches:
        return pd.DataFrame(columns=[
            "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score", "matched_via"
        ])

    rows = []
    for matched_text, score, _ in matches:
        matched_rows = df_site[df_site["supplier_name_fuzzy"] == matched_text].copy()
        if matched_rows.empty:
            continue

        for _, r in matched_rows[["seid", "Supplier_Id", "Supplier_Name"]].drop_duplicates().iterrows():
            rows.append({
                "Supplier_Id": r["Supplier_Id"],
                "Supplier_Name": r["Supplier_Name"],
                "exact_match": 0.0,
                "fuzzy_score": float(score),
                "cosine_score": 0.0,
                "matched_via": "fuzzy"
            })

    if not rows:
        return pd.DataFrame(columns=[
            "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score", "matched_via"
        ])

    return pd.DataFrame(rows)


def get_semantic_supplier_candidates(
    seid: int,
    query_semantic: str,
    model: SentenceTransformer,
    top_k: int = TOP_K_SEMANTIC
) -> pd.DataFrame:
    """
    FAISS semantic retrieval using supplier_name index + metadata.
    Assumes embeddings are normalized and index is IndexFlatIP.
    """
    if not query_semantic:
        return pd.DataFrame(columns=[
            "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score", "matched_via"
        ])

    index, metadata_df = load_supplier_resources(seid)

    if metadata_df.empty or index.ntotal == 0:
        return pd.DataFrame(columns=[
            "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score", "matched_via"
        ])

    query_vec = model.encode(
        [query_semantic],
        task="text-matching",
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    rows = []
    for score, idx in zip(scores[0], indices[0]):
        # print("\nINDEX:", idx)
        if idx < 0 or idx >= len(metadata_df):
            continue

        meta_row = metadata_df.iloc[int(idx)]

        supplier_id = meta_row.get("Supplier_Id", None)
        supplier_name = meta_row.get("Supplier_Name", None)

        if pd.isna(supplier_id) or pd.isna(supplier_name):
            continue

        rows.append({
            "seid": seid,
            "Supplier_Id": supplier_id,
            "Supplier_Name": str(supplier_name),
            "exact_match": 0.0,
            "fuzzy_score": 0.0,
            "cosine_score": float(score),
            "matched_via": "semantic"
        })

    # print("Query:", query_semantic)
    # print("Index size:", index.ntotal)
    # print("Metadata size:", len(metadata_df))
    # print("Top scores:", scores)
    # print("Top indices:", indices)

    if not rows:
        return pd.DataFrame(columns=[
            "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score", "matched_via"
        ])

    return pd.DataFrame(rows)


# =========================================================
# BACKFILL MISSING SCORES
# =========================================================

def backfill_candidate_scores(
    grouped: pd.DataFrame,
    query_fuzzy: str,
    query_semantic: str,
    model: SentenceTransformer = None
) -> pd.DataFrame:
    if grouped.empty:
        return grouped

    needs_fuzzy = (grouped["fuzzy_score"] == 0.0) & (grouped["exact_match"] == 0.0)
    if needs_fuzzy.any() and query_fuzzy:
        for idx in grouped[needs_fuzzy].index:
            candidate_fuzzy = normalize_supplier_name_fuzzy(grouped.at[idx, "Supplier_Name"])
            if candidate_fuzzy:
                score = fuzz.WRatio(query_fuzzy, candidate_fuzzy)
                grouped.at[idx, "fuzzy_score"] = float(score)

    needs_cosine = (grouped["cosine_score"] == 0.0) & (grouped["exact_match"] == 0.0)
    if needs_cosine.any() and query_semantic and model is not None:
        query_vec = model.encode(
            [query_semantic],
            task="text-matching",
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        candidate_indices = grouped[needs_cosine].index.tolist()
        candidate_texts = [
            normalize_supplier_name_semantic(grouped.at[idx, "Supplier_Name"])
            for idx in candidate_indices
        ]

        valid = [(idx, text) for idx, text in zip(candidate_indices, candidate_texts) if text]
        if valid:
            valid_indices, valid_texts = zip(*valid)
            candidate_vecs = model.encode(
                list(valid_texts),
                task="text-matching",
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype(np.float32)

            scores = (candidate_vecs @ query_vec.T).flatten()
            for idx, score in zip(valid_indices, scores):
                grouped.at[idx, "cosine_score"] = float(score)

    return grouped


# =========================================================
# MERGE + RANK
# =========================================================

def merge_and_rank_candidates(
    exact_df: pd.DataFrame,
    fuzzy_df: pd.DataFrame,
    semantic_df: pd.DataFrame,
    top_k_final: int = TOP_K_FINAL,
    weights: Optional[Dict[str, float]] = None,
    query_fuzzy: str = "",
    query_semantic: str = "",
    model: SentenceTransformer = None
) -> pd.DataFrame:
    """
    Merge candidates by Supplier_Id and compute weighted final score.
    """
    if weights is None:
        weights = WEIGHTS

    all_candidates = pd.concat(
        [
            exact_df,
            fuzzy_df,
            semantic_df
        ],
        ignore_index=True,
        sort=False
    )

    if all_candidates.empty:
        return pd.DataFrame(columns=[
            "seid", "Supplier_Id", "Supplier_Name", "exact_match", "fuzzy_score", "cosine_score",
            "final_score", "matched_via"
        ])

    # Normalize score columns
    all_candidates["exact_match"] = pd.to_numeric(all_candidates["exact_match"], errors="coerce").fillna(0.0)
    all_candidates["fuzzy_score"] = pd.to_numeric(all_candidates["fuzzy_score"], errors="coerce").fillna(0.0)
    all_candidates["cosine_score"] = pd.to_numeric(all_candidates["cosine_score"], errors="coerce").fillna(0.0)

    # Group duplicates by Supplier_Id
    grouped = (
        all_candidates
        .groupby("Supplier_Id", as_index=False)
        .agg({
            # "seid": "first",
            "Supplier_Name": "first",
            "exact_match": "max",
            "fuzzy_score": "max",
            "cosine_score": "max",
            "matched_via": lambda s: ",".join(sorted(set([x for x in s if pd.notna(x) and str(x).strip() != ""])))
        })
    )

    grouped = backfill_candidate_scores(
        grouped, query_fuzzy, query_semantic, model
    )

    grouped["fuzzy_score_norm"] = grouped["fuzzy_score"] / 100.0

    grouped["final_score"] = (
        weights["exact"] * grouped["exact_match"] +
        weights["fuzzy"] * grouped["fuzzy_score_norm"] +
        weights["semantic"] * grouped["cosine_score"]
    )

    grouped = grouped.sort_values(
        by=["final_score", "exact_match", "fuzzy_score", "cosine_score"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    grouped["rank"] = np.arange(1, len(grouped) + 1)

    return grouped.head(top_k_final)


# =========================================================
# LLM RERANKING
# =========================================================
RERANK_PROMPT = BASE_DIR / "Prompt" / "supplier_rerank.yaml"
with open(RERANK_PROMPT, encoding="utf-8") as f:
    prompt = yaml.safe_load(f)
RERANK_SYSTEM_PROMPT = prompt["system_prompt"]

SCHEMA_FILE = BASE_DIR / "Response_Schema" / "supplier_rerank_schema.json"
with open(SCHEMA_FILE, "r") as f:
    RERANK_RESPONSE_SCHEMA = json.load(f)


def get_sample_parts_for_candidate(
    df_site: pd.DataFrame,
    supplier_id: Any,
    n_parts: int = PARTS_PER_CANDIDATE
) -> list:
    if "PartName_Descriptive" not in df_site.columns:
        return []

    rows = df_site[df_site["Supplier_Id"] == supplier_id]
    rows = rows[rows["PartName_Descriptive"].notna()]
    rows = rows.drop_duplicates(subset=["PartName_Descriptive"])

    if rows.empty:
        return []

    parts = rows["PartName_Descriptive"].astype(str).head(n_parts).tolist()
    return parts


def rerank_candidates_with_llm(
    ranked_df: pd.DataFrame,
    invoice_json: Dict[str, Any],
    df_site: pd.DataFrame,
    top_k_rerank: int = TOP_K_RERANK,
    n_parts_per_candidate: int = PARTS_PER_CANDIDATE
) -> Dict[str, Any]:

    extracted_supplier = invoice_json.get("supplier_name", "")
    line_items = invoice_json.get("line_items", [])

    if line_items:
        line_items_text = "\n".join(
            f"- {item.get('item_name', 'N/A')} "
            f"(qty: {item.get('quantity', 'N/A')}, "
            f"rate: {item.get('rate', 'N/A')}, "
            f"amount: {item.get('amount', 'N/A')})"
            for item in line_items
        )
    else:
        line_items_text = "No line items available."

    candidates_text = ""
    for _, row in ranked_df.iterrows():
        sample_parts = get_sample_parts_for_candidate(
            df_site=df_site,
            supplier_id=row.get("Supplier_Id"),
            n_parts=n_parts_per_candidate
        )

        if sample_parts:
            parts_block = "\n".join(f"    - {p}" for p in sample_parts)
        else:
            parts_block = "    - (no part data available)"

        candidates_text += (
            f"Rank {int(row.get('rank', 0))}: "
            f"Supplier_Id={row.get('Supplier_Id', '')}, "
            f"Name=\"{row.get('Supplier_Name', '')}\", "
            f"Final_Score={row.get('final_score', 0.0):.4f}, "
            f"Fuzzy_Score={row.get('fuzzy_score', 0.0):.1f}, "
            f"Cosine_Score={row.get('cosine_score', 0.0):.4f}, "
            f"Matched_Via={row.get('matched_via', '')}\n"
            f"  Sample products/services from database:\n{parts_block}\n"
        )

    user_message = (
        f"Extracted supplier name from invoice: \"{extracted_supplier}\"\n\n"
        f"Invoice line items:\n{line_items_text}\n\n"
        f"Candidate matches from database:\n{candidates_text}\n"
        f"Rerank these candidates and return the top {top_k_rerank} most likely correct matches."
    )

    client = _get_bedrock_client()

    request_payload = {
    "modelId": BEDROCK_CONFIG["api"]["model_name"],
    "system": build_system_blocks(RERANK_SYSTEM_PROMPT, BEDROCK_CONFIG),
    "messages": [
        {
            "role": "user",
            "content": [{"text": user_message}]
        }
    ],
    "outputConfig": {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "name": "supplier_reranking",
                    "schema": json.dumps(RERANK_RESPONSE_SCHEMA)
                }
            }
        }
    },
    "inferenceConfig": {
        "temperature": BEDROCK_CONFIG["api"]["temperature"],
        "maxTokens": BEDROCK_CONFIG["api"]["max_tokens"]
    }
    }
    request_to_save = json.loads(json.dumps(request_payload, default=str))
    with open(DEBUG_DIR / f"{timestamp}_request.json", "w", encoding="utf-8") as f:
        json.dump(request_to_save, f, indent=4, ensure_ascii=False)

    response = client.converse(**request_payload)

    with open(DEBUG_DIR / f"{timestamp}_response_raw.json", "w", encoding="utf-8") as f:
        json.dump(response, f, indent=4, default=str, ensure_ascii=False)

    llm_result = parse_model_json(response)

    reranked = []
    valid_ranks = set(int(r) for r in ranked_df["rank"].tolist())
    for item in llm_result.get("reranked_candidates", [])[:top_k_rerank]:
        rank_raw = item.get("original_rank")
        try:
            original_rank = int(rank_raw)
        except (TypeError, ValueError):
            print(f"Skipping rerank candidate with non-integer original_rank: {rank_raw!r}")
            continue
        if original_rank not in valid_ranks:
            print(f"Skipping rerank candidate with out-of-range original_rank: {original_rank}")
            continue

        match_row = ranked_df[ranked_df["rank"] == original_rank]
        if match_row.empty:
            continue

        row = match_row.iloc[0]
        reranked.append({
            "Supplier_Id": row["Supplier_Id"],
            "Supplier_Name": row["Supplier_Name"],
            "final_score": float(row["final_score"]),
            "confidence": item.get("confidence", "low"),
            "reason": item.get("reason", ""),
            "original_rank": original_rank
        })

    return {
        "reranked_candidates": reranked,
        "no_match": llm_result.get("no_match", False),
        "no_match_reason": llm_result.get("no_match_reason", "")
    }


# =========================================================
# MAIN SUPPLIER MAPPING FUNCTION
# =========================================================

def map_supplier_name_from_invoice(
    invoice_json: Dict[str, Any],
    df_master: pd.DataFrame,
    model: SentenceTransformer,
    top_k_fuzzy: int = TOP_K_FUZZY,
    top_k_semantic: int = TOP_K_SEMANTIC,
    fuzzy_cutoff: int = FUZZY_CUTOFF,
    top_k_final: int = TOP_K_FINAL
) -> Dict[str, Any]:
    """
    Returns:
      - best_supplier_id
      - best_supplier_name
      - best_score
      - ranked_candidates (DataFrame)
    """
    site_id = invoice_json.get("Site_Id", None)
    extracted_supplier = invoice_json.get("supplier_name", "")

    if site_id is None:
        raise ValueError("Invoice JSON does not contain Site_Id.")

    try:
        site_id = int(site_id)
    except Exception as e:
        raise ValueError(f"Site_Id must be convertible to int. Got: {site_id}") from e

    # Step 1: filter master by seid
    df_site = prepare_supplier_site_df(df_master, site_id)

    if df_site.empty:
        return {
            "best_supplier_id": None,
            "best_supplier_name": None,
            "best_score": 0.0,
            "ranked_candidates": pd.DataFrame(),
            "query": {
                "site_id": site_id,
                "extracted_supplier": extracted_supplier
            }
        }

    # Step 2: normalize extracted supplier name twice
    supplier_fuzzy_query = normalize_supplier_name_fuzzy(extracted_supplier)
    supplier_semantic_query = normalize_supplier_name_semantic(extracted_supplier)

    # Step 3: exact / fuzzy / semantic candidates
    exact_df = get_exact_supplier_candidates(df_site, supplier_fuzzy_query)
    fuzzy_df = get_fuzzy_supplier_candidates(
        df_site=df_site,
        query_fuzzy=supplier_fuzzy_query,
        top_k=top_k_fuzzy,
        cutoff=fuzzy_cutoff
    )
    semantic_df = get_semantic_supplier_candidates(
        seid=site_id,
        query_semantic=supplier_semantic_query,
        model=model,
        top_k=top_k_semantic
    )

    print("SEMANTIC DF:")
    print(semantic_df.to_string(index=False))

    # Step 4: merge and rank
    ranked_df = merge_and_rank_candidates(
        exact_df=exact_df,
        fuzzy_df=fuzzy_df,
        semantic_df=semantic_df,
        top_k_final=top_k_final,
        query_fuzzy=supplier_fuzzy_query,
        query_semantic=supplier_semantic_query,
        model=model
    )

    if ranked_df.empty:
        return {
            "best_supplier_id": None,
            "best_supplier_name": None,
            "best_score": 0.0,
            "ranked_candidates": ranked_df,
            "query": {
                "site_id": site_id,
                "extracted_supplier": extracted_supplier,
                "supplier_fuzzy_query": supplier_fuzzy_query,
                "supplier_semantic_query": supplier_semantic_query
            }
        }

    best_row = ranked_df.iloc[0]

    result = {
        "best_supplier_id": best_row["Supplier_Id"],
        "best_supplier_name": best_row["Supplier_Name"],
        "best_score": float(best_row["final_score"]),
        "ranked_candidates": ranked_df,
        "llm_reranked": False,
        "llm_reranked_candidates": [],
        "no_match": False,
        "no_match_reason": "",
        "query": {
            "site_id": site_id,
            "extracted_supplier": extracted_supplier,
            "supplier_fuzzy_query": supplier_fuzzy_query,
            "supplier_semantic_query": supplier_semantic_query
        }
    }
    print(f"AAA --> {result['best_score']}")

    # High-confidence numeric auto-accept: if the top candidate is not a byte-exact match
    # but BOTH its fuzzy and semantic scores are very high, the match is unambiguous enough
    # to skip the LLM rerank entirely (saves one ~20s Bedrock call).
    best_fuzzy = float(best_row["fuzzy_score"])
    best_cosine = float(best_row["cosine_score"])
    is_exact = float(best_row["exact_match"]) >= 1.0
    high_conf_non_exact = (
        not is_exact
        and best_fuzzy >= SUPPLIER_AUTO_ACCEPT_FUZZY
        and best_cosine >= SUPPLIER_AUTO_ACCEPT_COSINE
    )

    if high_conf_non_exact:
        result["auto_accepted"] = True
        result["auto_accept_confidence"] = SUPPLIER_AUTO_ACCEPT_CONFIDENCE
        result["auto_accept_reason"] = (
            f"High-confidence name match to '{result['best_supplier_name']}' "
            f"(fuzzy {best_fuzzy:.0f}, cosine {best_cosine:.2f}); "
            f"auto-accepted without LLM rerank."
        )
    elif result["best_score"] < LLM_RERANK_THRESHOLD:
        try:
            print(f"1")
            rerank_result = rerank_candidates_with_llm(
                ranked_df=ranked_df,
                invoice_json=invoice_json,
                df_site=df_site,
                top_k_rerank=TOP_K_RERANK,
                n_parts_per_candidate=PARTS_PER_CANDIDATE
            )
            print(f"2")


            result["llm_reranked"] = True
            result["llm_reranked_candidates"] = rerank_result["reranked_candidates"]
            result["no_match"] = rerank_result["no_match"]
            result["no_match_reason"] = rerank_result["no_match_reason"]

            print(f"3")

            if rerank_result["reranked_candidates"] and not rerank_result["no_match"]:
                print(f"4")
                top_reranked = rerank_result["reranked_candidates"][0]
                result["best_supplier_id"] = top_reranked["Supplier_Id"]
                result["best_supplier_name"] = top_reranked["Supplier_Name"]
                result["best_score"] = top_reranked["final_score"]

        except Exception as e:
            print(f"LLM reranking failed, using original ranking: {e}")

    return result