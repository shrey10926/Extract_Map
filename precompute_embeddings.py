from pathlib import Path
import os, faiss, pandas as pd, numpy as np
import torch


# =========================================================
# CONFIG
# =========================================================

PARQUET_PATH = Path("data/master_dataset.parquet")

HF_MODEL_REPO = "jinaai/jina-embeddings-v5-text-small-text-matching"
LOCAL_MODEL_DIR = Path("./models/jina-embeddings-v5-text-small-text-matching")

OUTPUT_DIR = Path(r"./vector_store")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FAISS_DIR = OUTPUT_DIR / "faiss_indexes"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

METADATA_DIR = OUTPUT_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================================
# ENTITY CONFIG
# =========================================================
# Change these column names to match your dataframe.

ENTITY_CONFIG = {
    "supplier_name": {
        "text_col": "supplier_name_semantic",
        "metadata_cols": [
            "seid",
            "Supplier_Id",
            "Supplier_Name",
            "supplier_name_semantic"
        ]
    },
    "item_name": {
        "text_col": "item_name_semantic",
        "metadata_cols": [
            "seid",
            "Part_id",
            "PartName_Descriptive",
            "item_name_semantic",
            "Sup_Part_Code",
            "Reg_Price"
        ]
    }
}

# =========================================================
# ONE-TIME DOWNLOAD CHECK
# =========================================================

if not LOCAL_MODEL_DIR.exists():
    print(f"Local model folder not found at: {LOCAL_MODEL_DIR}")
    print(f"Downloading {HF_MODEL_REPO} to local directory...")

    from huggingface_hub import snapshot_download

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_MODEL_REPO,
        local_dir=str(LOCAL_MODEL_DIR),
        ignore_patterns=["*.gguf", "*.bin", "*.mlx"]
    )
    print("Download completed successfully.\n")
else:
    print(f"Using local model found at: {LOCAL_MODEL_DIR.resolve()}")

# =========================================================
# FORCE OFFLINE MODE
# =========================================================

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

# =========================================================
# LOAD MASTER DATASET
# =========================================================

print("Loading parquet dataset...")
df = pd.read_parquet(PARQUET_PATH)
print(f"Rows Loaded: {len(df)}")

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

print(f"\nLoading embedding model from local path: {LOCAL_MODEL_DIR}")
model = SentenceTransformer(
    str(LOCAL_MODEL_DIR),
    device=device,#"cuda"
    trust_remote_code=True
)
print("Embedding model loaded.")

# =========================================================
# UTILITY FUNCTION
# =========================================================

def build_faiss_for_column(df_site: pd.DataFrame, seid, entity_name: str, text_col: str, metadata_cols: list):
    """
    Build FAISS index + metadata parquet for one entity type.
    """

    print("\n=================================================")
    print(f"Processing seid: {seid} | entity: {entity_name}")
    print("=================================================")

    if text_col not in df_site.columns:
        print(f"Skipping {entity_name}: column '{text_col}' not found.")
        return

    # Keep only valid rows
    df_idx = df_site.copy()
    df_idx = df_idx[df_idx[text_col].notna()].copy()
    df_idx[text_col] = df_idx[text_col].astype(str).str.strip()
    df_idx = df_idx[df_idx[text_col] != ""].copy()

    print(f"Rows for {entity_name}: {len(df_idx)}")

    if len(df_idx) == 0:
        print(f"No valid rows found for {entity_name}. Skipping.")
        return

    # Optional: deduplicate on text column to reduce repeated vectors
    # Uncomment if your master data has many duplicates.
    # df_idx = df_idx.drop_duplicates(subset=[text_col]).copy()

    df_idx.reset_index(drop=False, inplace=True)
    df_idx.rename(columns={"index": "source_row_id"}, inplace=True)

    texts = df_idx[text_col].tolist()

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        device=device,
        task="text-matching",
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"Embedding shape: {embeddings.shape}")

    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    print(f"Vectors added to FAISS: {index.ntotal}")

    # Save FAISS index
    faiss_index_path = FAISS_DIR / f"faiss_seid_{seid}_{entity_name}.index"
    faiss.write_index(index, str(faiss_index_path))
    print(f"FAISS index saved: {faiss_index_path}")

    # Save metadata
    existing_cols = [c for c in metadata_cols if c in df_idx.columns]
    metadata_df = df_idx[existing_cols].copy()

    # Add useful tracking columns
    metadata_df["source_row_id"] = df_idx["source_row_id"].values
    metadata_df["vector_id"] = np.arange(len(metadata_df))
    metadata_df["entity_type"] = entity_name
    metadata_df["source_text"] = df_idx[text_col].values

    metadata_path = METADATA_DIR / f"metadata_seid_{seid}_{entity_name}.parquet"
    metadata_df.to_parquet(metadata_path, index=False)
    print(f"Metadata saved: {metadata_path}")

# =========================================================
# LOOP THROUGH EACH SEID
# =========================================================

unique_seids = df["seid"].dropna().unique().tolist()
print(f"\nUnique seid count: {len(unique_seids)}")

for seid in unique_seids:
    df_site = df[df["seid"] == seid].copy()
    df_site.reset_index(drop=True, inplace=True)

    print("\n=================================================")
    print(f"Processing seid: {seid}")
    print("Rows for seid:", len(df_site))
    print("=================================================")

    for entity_name, cfg in ENTITY_CONFIG.items():
        build_faiss_for_column(
            df_site=df_site,
            seid=seid,
            entity_name=entity_name,
            text_col=cfg["text_col"],
            metadata_cols=cfg["metadata_cols"]
        )

print("\n========================================")
print("ALL INDEXES CREATED SUCCESSFULLY")
print("========================================")