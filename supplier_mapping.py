from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import json, re, os, unicodedata, yaml, boto3, faiss, numpy as np, pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================

MASTER_PARQUET_PATH = Path("data\master_dataset.parquet")

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
LLM_RERANK_THRESHOLD = 0.85
TOP_K_RERANK = 5
PARTS_PER_CANDIDATE = 3

with open("bedrock_config.yaml", "r") as f:
    BEDROCK_CONFIG = yaml.safe_load(f)

_bedrock_client_cache = None


def _get_bedrock_client():
    global _bedrock_client_cache
    if _bedrock_client_cache is None:
        session = boto3.Session(profile_name="shrey_bedrock")
        _bedrock_client_cache = session.client(
            "bedrock-runtime", region_name="us-east-1"
        )
    return _bedrock_client_cache


# =========================================================
# NORMALIZATION
# =========================================================

def lowercase_text(text: str) -> str:
    if pd.isna(text):
        return ""
    return str(text).lower()


def trim_extra_spaces(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def unicode_normalization(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")
    return text


def safe_separator_normalization(text: str) -> str:
    """
    Safely normalize separators while preserving decimal numbers.

    Examples:
        1000BULBS.com -> 1000bulbs com
        REM/TOP       -> rem top
        3.5OZ         -> 3.5oz (preserved)
    """
    if pd.isna(text):
        return ""

    text = str(text)

    # Replace separators with spaces
    text = re.sub(r"[,_/\-]+", " ", text)

    # Replace dots NOT between digits
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_supplier_name_fuzzy(text: str) -> str:
    """
    supplier_name_fuzzy pipeline
    """
    text = lowercase_text(text)
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    text = safe_separator_normalization(text)
    text = trim_extra_spaces(text)
    return text


def normalize_supplier_name_semantic(text: str) -> str:
    """
    supplier_name_semantic pipeline
    """
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    text = trim_extra_spaces(text)
    return text


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


def load_invoice_json(invoice_input: Any) -> Dict[str, Any]:
    """
    Accepts either:
      - dict
      - JSON string
      - path to a JSON file
    """
    if isinstance(invoice_input, dict):
        return invoice_input

    if isinstance(invoice_input, str):
        s = invoice_input.strip()

        # File path
        if Path(s).exists():
            with open(s, "r", encoding="utf-8") as f:
                return json.load(f)

        # JSON string
        return json.loads(s)

    raise TypeError("invoice_input must be a dict, JSON string, or JSON file path.")


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
    
    # (1, 1024)
    # print(f"AAAAAAAAAAAAAAAAAAAAAAAAAA -->> {query_vec.shape}")

    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    rows = []
    print("Metadata size:", len(metadata_df))
    print(f"ZZZZZZZZ -->> {metadata_df.columns.tolist()}")
    print(metadata_df.head())

    for score, idx in zip(scores[0], indices[0]):
        print("\nINDEX:", idx)
        if idx < 0 or idx >= len(metadata_df):
            continue

        meta_row = metadata_df.iloc[int(idx)]

        print(f"META ROW -->> {meta_row}")

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

    print("Query:", query_semantic)
    print("Index size:", index.ntotal)
    print("Metadata size:", len(metadata_df))
    print("Top scores:", scores)
    print("Top indices:", indices)

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

RERANK_SYSTEM_PROMPT = """\
You are a supplier name matching expert. You are given:
1. A supplier name extracted from an invoice
2. Line items from the invoice showing what products/services this supplier provides
3. A list of candidate matches from a database, each with their matching scores AND a sample of products/services that candidate supplies according to the database

Rerank the candidates by likelihood of being the correct match.

Consider:
- String similarity between the extracted name and candidate names
- Whether the candidate's database products/services align with the invoice line items (this is a strong signal — a candidate whose sample products match the invoice line items is much more likely to be correct, even if the name similarity is weaker)
- Common supplier name variations (abbreviations, legal suffixes like Inc/LLC/Corp, DBA names, parent companies)
- Spelling errors or OCR artifacts in the extracted name

Note: The sample products are only a small subset of what the candidate supplies — a missing product category does not mean the candidate is wrong, but matching categories is a positive signal.

Return the top 5 most likely matches with confidence level and reasoning.
If none of the candidates appear to be a correct match, set no_match to true and explain why."""

RERANK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reranked_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original_rank": {
                        "type": "integer",
                        "description": "The rank number (1-10) of this candidate from the input list"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"]
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation for why this candidate is or isn't a good match"
                    }
                },
                "required": ["original_rank", "confidence", "reason"]
            }
        },
        "no_match": {
            "type": "boolean",
            "description": "True if none of the candidates appear to be a correct match"
        },
        "no_match_reason": {
            "type": "string",
            "description": "Explanation of why no candidates match, empty string if no_match is false"
        }
    },
    "required": ["reranked_candidates", "no_match", "no_match_reason"]
}


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

    response = client.converse(
        modelId=BEDROCK_CONFIG["api"]["model_name"],
        system=[{"text": RERANK_SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [{"text": user_message}]
            }
        ],
        outputConfig={
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
        inferenceConfig={
            "temperature": 0,
            "maxTokens": 1024
        }
    )

    response_text = (
        response["output"]["message"]["content"][0]["text"]
        .strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
    )

    llm_result = json.loads(response_text)

    reranked = []
    for item in llm_result.get("reranked_candidates", [])[:top_k_rerank]:
        original_rank = item.get("original_rank")
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

    if result["best_score"] < LLM_RERANK_THRESHOLD:
        try:
            rerank_result = rerank_candidates_with_llm(
                ranked_df=ranked_df,
                invoice_json=invoice_json,
                df_site=df_site,
                top_k_rerank=TOP_K_RERANK,
                n_parts_per_candidate=PARTS_PER_CANDIDATE
            )

            result["llm_reranked"] = True
            result["llm_reranked_candidates"] = rerank_result["reranked_candidates"]
            result["no_match"] = rerank_result["no_match"]
            result["no_match_reason"] = rerank_result["no_match_reason"]

            if rerank_result["reranked_candidates"] and not rerank_result["no_match"]:
                top_reranked = rerank_result["reranked_candidates"][0]
                result["best_supplier_id"] = top_reranked["Supplier_Id"]
                result["best_supplier_name"] = top_reranked["Supplier_Name"]
                result["best_score"] = top_reranked["final_score"]

        except Exception as e:
            print(f"LLM reranking failed, using original ranking: {e}")

    return result


# =========================================================
# EXAMPLE USAGE
# =========================================================

if __name__ == "__main__":
    # Example invoice JSON
    invoice_json = {
        "Site_Id": 11,
    "supplier_name": "Republic Services",
    "invoice_date": "February 28, 2028",
    "invoice_number": "0918-006708380",
    "line_items": [
        {
            "item_name": "Waste Container 40 Yd, On Call Service",
            "quantity": "1.0000",
            "rate": "174.08",
            "amount": "174.08"
        },
        {
            "item_name": "Waste Container 40 Yd, On Call Service",
            "quantity": "1.0000",
            "rate": "122.70",
            "amount": "122.70"
        },
        {
            "item_name": "Waste Container 40 Yd, On Call Service",
            "quantity": "1.0000",
            "rate": "1646.56",
            "amount": "1646.56"
        },
        {
            "item_name": "Waste Container 40 Yd, On Call Service",
            "quantity": "1.0000",
            "rate": "174.08",
            "amount": "174.08"
        },
        {
            "item_name": "Waste Container 40 Yd, On Call Service",
            "quantity": "1.0000",
            "rate": "1646.56",
            "amount": "1646.56"
        },
        {
            "item_name": "Waste Container 40 Yd, On Call Service",
            "quantity": "1.0000",
            "rate": "1646.56",
            "amount": "1646.56"
        }
    ],
    "total_invoice_amount": "5287.84"
}

    # Load data/model
    master_df = load_master_dataframe()
    model = load_embedding_model()

    result = map_supplier_name_from_invoice(
        invoice_json=invoice_json,
        df_master=master_df,
        model=model,
        top_k_fuzzy=10,
        top_k_semantic=10,
        fuzzy_cutoff=80,
        top_k_final=10
    )

    print("\nQUERY:")
    print(result["query"])

    print("\nBEST MATCH:")
    print("Supplier_Id  :", result["best_supplier_id"])
    print("Supplier_Name:", result["best_supplier_name"])
    print("Score        :", result["best_score"])

    print("\nRANKED CANDIDATES:")
    print(result["ranked_candidates"].to_string(index=False))

    print("\nLLM RERANKED:", result["llm_reranked"])
    if result["llm_reranked"]:
        print("No Match:", result["no_match"])
        if result["no_match"]:
            print("Reason:", result["no_match_reason"])
        print("\nRERANKED CANDIDATES:")
        for c in result["llm_reranked_candidates"]:
            print(
                f"  Original Rank {c['original_rank']}: "
                f"{c['Supplier_Name']} (Id={c['Supplier_Id']}) "
                f"| Confidence: {c['confidence']} "
                f"| Score: {c['final_score']:.4f} "
                f"| Reason: {c['reason']}"
            )
