from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
import json, re, os, unicodedata, faiss, numpy as np, pandas as pd
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
# MERGE + RANK
# =========================================================

def merge_and_rank_candidates(
    exact_df: pd.DataFrame,
    fuzzy_df: pd.DataFrame,
    semantic_df: pd.DataFrame,
    top_k_final: int = TOP_K_FINAL,
    weights: Optional[Dict[str, float]] = None
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
        top_k_final=top_k_final
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

    return {
        "best_supplier_id": best_row["Supplier_Id"],
        "best_supplier_name": best_row["Supplier_Name"],
        "best_score": float(best_row["final_score"]),
        "ranked_candidates": ranked_df,
        "query": {
            "site_id": site_id,
            "extracted_supplier": extracted_supplier,
            "supplier_fuzzy_query": supplier_fuzzy_query,
            "supplier_semantic_query": supplier_semantic_query
        }
    }


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
