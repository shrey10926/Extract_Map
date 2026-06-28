# Technical Requirements Document — Invoice Entity Extraction & Mapping System

**Version:** 2.0 (self-contained build spec)
**Audience:** A coding agent that will implement the system from this document.
**Owner:** sgirishkumarjain@deloitte.com

> **How to read this document.** This spec is written so you can build the system without
> any other explanation. Section 3 (Glossary) defines every term used later — read it first.
> Section 4 lists what is **already provided** to you (do not recreate these) versus what you
> must **build**. Sections 7–10 specify, function by function, exactly what to implement.

---

## 1. Purpose

Given an request json containing Site_Id and S3 bucket URL, the system:
1. **Downloads and Extracts** structured fields (entities) from the invoice using a vision LLM.
2. **Maps the supplier name** on the invoice to a supplier record in a catalog database.
3. **Maps each line item** on the invoice to a part record belonging to that matched supplier.
4. Returns a single structured JSON result.

"Mapping" means: convert free-text extracted by the LLM into the database's canonical
identifiers (`Supplier_Id`, `Part_id`) so an invoice can be reconciled against the catalog.

---

## 2. Scope

**In scope:** entity extraction; supplier mapping; line-item mapping (scoped to the matched
supplier); one orchestrator script producing one JSON result per file.
- Downloading the invoice from the request's `signedUrl`.
- A FastAPI API which accepts a request json containing Site_Id and S3 bucker URL for the invoice and gives the structured JSON result.

**Out of scope (this version):**
- Multiple invoices inside one file (the output structure allows it, but exactly one is produced).
- Using price / quantity / amount to decide matches (matching is **name-only**; those values are only copied into the output).

---

## 3. Glossary — definitions of every term used in this document

- **Entity** — a field extracted from the invoice. The entities are: `supplier_name`,
  `invoice_date`, `invoice_number`, `line_items` (a list), and `total_invoice_amount`. Each
  line item has `item_name`, `quantity`, `rate`, `amount`.
- **Site / `seid` / `Site_Id` / `siteid`** — an integer identifying a physical site. The catalog
  is partitioned by site; mapping only ever searches within one site. The **same integer**
  appears under three names: `siteid` (in the input request), `seid` (column in the catalog
  table and vector-store files), and `Site_Id` (key inside the invoice JSON handed to the
  mapping code). Treat them as identical.
- **master_dataset** — the catalog table, stored at `data/master_dataset.parquet`. It is a flat
  table where **each row is one (supplier, part) combination**. It is produced by the provided
  `export_parquet.py`. Its columns are listed in Section 6.1. The whole mapping system searches
  inside this table. In code it is loaded into a pandas DataFrame usually called `df_master`.
- **Candidate** — a database row that *might* be the correct match for an extracted entity. The
  system produces several candidates per entity and scores them.
- **Normalization** — converting raw text into a canonical comparable form before matching.
  There are **two** normalized forms (see Section 6.3): a **fuzzy form** (for string comparison)
  and a **semantic form** (for embeddings). The exact steps are specified; reproduce them
  precisely or matches will be wrong.
- **Embedding** — a fixed-length numeric vector (list of floats) representing the *meaning* of a
  piece of text, produced by the sentence-transformer model. Texts with similar meaning produce
  vectors pointing in similar directions. All embeddings here are **L2-normalized** (unit length).
- **Cosine similarity** — a number measuring how aligned two vectors are. Because the vectors are
  unit length, cosine similarity equals their **dot product**. Range here ≈ 0…1; 1 means
  identical meaning.
- **FAISS index** — a binary file storing many embeddings for fast similarity lookup. The indexes
  here are of type `IndexFlatIP` ("flat, inner product"), which stores the raw vectors and
  compares a query against **all** of them exactly. For unit vectors, inner product = cosine.
- **Reconstruct** — reading the raw stored vectors back out of a FAISS index
  (`index.reconstruct_n(0, index.ntotal)`). This is a memory copy; it does **not** run the model.
  Used to reuse precomputed embeddings without re-encoding.
- **Exact match** — the *fuzzy-normalized* query string is byte-for-byte equal to a candidate's
  *fuzzy-normalized* string. Contributes a score of **1** (else 0).
- **Fuzzy match** — approximate string similarity using RapidFuzz's `WRatio` scorer, returning a
  number **0–100**. Tolerant of typos, word reordering, and OCR noise.
- **Semantic match** — similarity of *meaning* via embeddings + cosine similarity, returning
  **0–1**.
- **Component scores** — the three raw scores for a candidate: `exact_match` (0 or 1),
  `fuzzy_score` (0–100), `cosine_score` (0–1).
- **Weighted score / `final_score`** — a single combined score per candidate, in **0–1**,
  computed as `0.50·exact_match + 0.30·(fuzzy_score/100) + 0.20·cosine_score`. Higher is better.
- **Backfill** — a candidate found by one method is missing the other two component scores. For
  example, a candidate found only by fuzzy matching has `cosine_score = 0` even though it may be
  semantically similar. **Backfilling** computes the missing component scores for each candidate
  so that `final_score` is computed fairly from all three signals.
- **Rank** — candidates sorted by `final_score` descending; rank 1 is the best.
- **LLM rerank** — when the deterministic match is not certain, send the top candidates plus extra
  context to the LLM and ask it to choose/re-order. Returns a chosen candidate (by its rank or
  index) with a confidence level and a reason, or a "no match" flag.
- **Threshold** — the `final_score` cutoff that decides whether to auto-accept the deterministic
  result or invoke the LLM rerank.
- **`matched_via`** — a string label recording which method(s) produced a candidate
  (`"exact"`, `"fuzzy"`, `"semantic"`, or a comma-joined combination).

---

## 4. What is PROVIDED vs. what you must BUILD

**Provided to you as-is (do NOT recreate; consume them):**
- `bedrock_config.yaml` — LLM settings (Section 9).
- `Prompt/prompt.yaml` — extraction system prompt.
- `Prompt/rerank.yaml` — supplier rerank system prompt.
- `Prompt/item_rerank.yaml` — item rerank system prompt.
- `Response_Schema/response_schema_updated.json` — extraction output JSON schema.
- `Response_Schema/rerank_schema.json` — supplier rerank output JSON schema.
- `Response_Schema/item_rerank_schema.json` — item rerank output JSON schema.
- `export_parquet.py` — builds `data/master_dataset.parquet` from SQL Server. **Defines the
  authoritative text-normalization functions** (Section 6.3).
- `precompute_embeddings.py` — builds the FAISS indexes + metadata (Section 6.2).
- `data/master_dataset.parquet` and the `vector_store/` directory are assumed already built by
  running the two provided scripts.

**You must build (from this document):**
- `extract_entities.py` (Section 7)
- `supplier_mapping.py` (Section 8)
- `item_mapping.py` (Section 9)
- `main.py` (Section 10)
- FastAPI API which takes the request jsona dn gives the response json

---

## 5. Tech Stack

| Purpose | Library / Service | Notes |
|---|---|---|
| Language | Python 3.10+ | `str | None` union syntax is used |
| Vision + text LLM | AWS Bedrock, Claude Haiku 4.5 (inference-profile ARN) | called via `boto3` `bedrock-runtime` `converse` API |
| Embeddings | `sentence-transformers`, model `jinaai/jina-embeddings-v5-text-small-text-matching` | `encode(..., task="text-matching", normalize_embeddings=True)` |
| Vector search | `faiss` (`IndexFlatIP`) | exact inner-product; cosine because vectors are normalized |
| Fuzzy strings | `rapidfuzz` (`fuzz.WRatio`, `process.extract`, `process.cdist`) | scores 0–100 |
| Tables | `pandas`, `numpy`, `pyarrow` | Parquet I/O |
| Documents | `PyMuPDF` (`import fitz`), `Pillow` (`PIL`) | PDF render, image preprocessing |
| Config | `PyYAML` | reads the YAML files |
| Cloud auth | AWS SSO: `aws sso login --profile shrey_bedrock`, region `us-east-1` | required before running |

---

## 6. Data Contracts of Provided Artifacts

### 6.1 `data/master_dataset.parquet` (the catalog; from `export_parquet.py`)
One row per (supplier, part). Columns you will use:

| Column | Type | Meaning |
|---|---|---|
| `seid` | Int64 | site id — filter key |
| `Supplier_Id` | Int64 | supplier identifier (a mapping output) |
| `Supplier_Name` | str | supplier display name |
| `Part_id` | Int64 | part identifier (a mapping output) |
| `PartName_Descriptive` | str | part display name; the source text for item embeddings |
| `Sup_Part_Code` | str | supplier's code for the part reference only — **never used in matching** |
| `Reg_Price`, `Min_Qty` | numeric | reference only — **never used in matching** |
| `supplier_name_fuzzy` | str | supplier name pre-normalized to **fuzzy** form |
| `supplier_name_semantic` | str | supplier name pre-normalized to **semantic** form |
| `item_name_fuzzy` | str | part name pre-normalized to **fuzzy** form |
| `item_name_semantic` | str | part name pre-normalized to **semantic** form |

### 6.2 Vector store (from `precompute_embeddings.py`)
For **each** `seid` and **each** entity type (`supplier_name`, `item_name`) two files exist:

- FAISS index: `vector_store/faiss_indexes/faiss_seid_{seid}_{entity}.index`
  (type `IndexFlatIP`, vectors L2-normalized).
- Metadata table: `vector_store/metadata/metadata_seid_{seid}_{entity}.parquet`.

**Critical alignment guarantee:** metadata row `i` corresponds to vector `i` in the index
(`vector_id` column = row order). So reconstructing the index gives vectors aligned 1:1 with the
metadata rows.

Metadata columns:
- supplier_name: `seid, Supplier_Id, Supplier_Name, supplier_name_semantic`, plus
  `source_row_id, vector_id, entity_type, source_text`.
- item_name: `seid, Part_id, PartName_Descriptive, item_name_semantic, Sup_Part_Code, Reg_Price`,
  plus `source_row_id, vector_id, entity_type, source_text`.

`source_text` is the exact text that was embedded — i.e. the value of `supplier_name_semantic`
(or `item_name_semantic`), stripped. Two rows with the same `source_text` have the same vector
(an embedding is a pure function of its input text).

> **Note:** the item metadata does **not** contain `Supplier_Id`. To restrict item candidates to
> one supplier you will filter `master_dataset` by `Supplier_Id` and obtain each candidate's
> vector by matching its `item_name_semantic` text against `source_text` (Section 9.4).

### 6.3 Text normalization (authoritative; defined in provided `export_parquet.py`)
You must normalize query text at runtime **identically** to how these columns were built, per
entity. The pipelines are built from these primitive steps:

- `lowercase(t)` → `t.lower()`.
- `trim(t)` → collapse all whitespace runs to a single space (`re.sub(r"\s+", " ", t)`) then strip.
- `unicode(t)` → `unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("utf-8")`
  (drops accents/non-ASCII).
- `separators(t)` → replace any run of `, _ / -` with a space (`re.sub(r"[,_/\-]+", " ", t)`);
  replace a dot that is **not** between two digits with a space (`re.sub(r"(?<!\d)\.(?!\d)", " ", t)`,
  which preserves decimals like `3.5`); collapse spaces; strip.

Pipelines:

| Pipeline | Steps (in order) | Used for |
|---|---|---|
| **fuzzy** (supplier & item) | lowercase → trim → unicode → separators → trim | exact & fuzzy matching |
| **supplier semantic** | trim → unicode → trim  *(NO lowercase)* | supplier embeddings |
| **item semantic** | lowercase → trim → unicode → trim  *(WITH lowercase)* | item embeddings |

> ⚠️ The supplier and item **semantic** pipelines differ: item lowercases, supplier does not.
> Implement them separately. A `pd.isna`/None input must yield `""`.

### 6.4 LLM call contract (AWS Bedrock `converse`)
All LLM calls use the same shape. `BEDROCK_CONFIG = yaml.safe_load(open("bedrock_config.yaml"))`.

```python
client = boto3.Session(profile_name="shrey_bedrock").client("bedrock-runtime", region_name="us-east-1")
response = client.converse(
    modelId=BEDROCK_CONFIG["api"]["model_name"],
    system=[{"text": SYSTEM_PROMPT}],            # the relevant prompt's "system_prompt"
    messages=[{"role": "user", "content": CONTENT}],   # see below
    outputConfig={"textFormat": {"type": "json_schema", "structure": {
        "jsonSchema": {"name": NAME, "schema": json.dumps(SCHEMA_DICT)}}}},
    inferenceConfig={"temperature": BEDROCK_CONFIG["api"]["temperature"],
                     "maxTokens": BEDROCK_CONFIG["api"]["max_tokens"]},
)
text = response["output"]["message"]["content"][0]["text"]
text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
result = json.loads(text)   # always strip code fences before parsing
```

- **Extraction:** `CONTENT` is a list of image blocks
  `{"image": {"format": "png", "source": {"bytes": <png bytes>}}}`, one per page;
  `NAME="invoice_extraction"`, `SCHEMA_DICT` = the extraction schema.
- **Supplier rerank:** `CONTENT=[{"text": user_message}]`, `NAME="supplier_reranking"`,
  `SCHEMA_DICT` = supplier rerank schema.
- **Item rerank:** `CONTENT=[{"text": user_message}]`, `NAME="item_reranking"`,
  `SCHEMA_DICT` = item rerank schema.

The provided JSON schemas define the LLM output shape:
- Supplier rerank output: `{"reranked_candidates": [{"original_rank": int, "confidence":
  "high|medium|low", "reason": str}], "no_match": bool, "no_match_reason": str}`.
- Item rerank output: `{"items": [{"item_index": int, "best_candidate_rank": int|null,
  "confidence": "high|medium|low", "no_match": bool, "reason": str}]}`.

---

## 7. Component: `extract_entities.py` (BUILD)

**Goal:** turn a document file into extracted entities.

**Module setup:** load `bedrock_config.yaml`, the extraction prompt (`Prompt/prompt.yaml`,
key `system_prompt`), and the extraction schema (`Response_Schema/response_schema_updated.json`).
Create the Bedrock client (profile `shrey_bedrock`, region `us-east-1`).

**Functions to implement:**

1. `preprocess_image(img, output_path=None, max_edge=1568) -> bytes`
   - Convert image to RGB. If the longest edge > `max_edge`, resize proportionally so the longest
     edge equals `max_edge` (use Lanczos resampling). Encode as PNG and return the PNG bytes.
     Optionally also write the PNG to `output_path`.

2. `process_pdf(file_path, password=None) -> list[dict]`
   - Open with `fitz`. If encrypted, authenticate with `password` (raise a clear error if missing
     or wrong). Render each page at **2.5× zoom** to an image, run `preprocess_image`. Return a
     list of `{"page_number", "png_bytes", "saved_path"}`.

3. `process_tiff(file_path) -> list[dict]` — iterate all frames of a multi-frame TIFF; same output.

4. `process_image_file(file_path) -> list[dict]` — single-page for JPG/JPEG/PNG/WebP/BMP.

5. `convert_document(file_path, password=None) -> list[dict]`
   - Dispatch by file extension: `.pdf`→`process_pdf`; `.tif/.tiff`→`process_tiff`;
     `.jpg/.jpeg/.png/.webp/.bmp`→`process_image_file`; otherwise raise an "unsupported type" error.

6. `build_content_blocks(pages) -> list[dict]`
   - Map each page to `{"image": {"format": "png", "source": {"bytes": page["png_bytes"]}}}`.

7. `extract_invoice(file_path, password=None) -> dict`
   - `pages = convert_document(...)`; `content = build_content_blocks(pages)`.
   - Call Bedrock per Section 6.4 (extraction variant) with the extraction prompt + schema.
   - Parse the JSON (strip code fences). This parsed dict is the **entities** object whose shape
     is defined by the extraction schema: `{supplier_name, invoice_date, invoice_number,
     line_items:[{item_name, quantity, rate, amount}], total_invoice_amount}` (all values are
     strings or `null`).
   - **Return** `{"file": Path(file_path).name, "total_pages": len(pages), "entities": <parsed>}`.

**Rules:** missing entities are already returned as `null` by the prompt+schema — do not invent
defaults. `temperature=0` (from config) for deterministic output.

---

## 8. Component: `supplier_mapping.py` (BUILD)

This module also hosts shared utilities reused by `item_mapping.py`.

### 8.1 Module config (constants)
```
MASTER_PARQUET_PATH = data/master_dataset.parquet
VECTOR_STORE_DIR    = vector_store
FAISS_DIR           = vector_store/faiss_indexes
METADATA_DIR        = vector_store/metadata
LOCAL_MODEL_DIR     = ./models/jina-embeddings-v5-text-small-text-matching
TOP_K_FUZZY = 10            # max candidates from fuzzy matching
TOP_K_SEMANTIC = 10         # max candidates from FAISS
FUZZY_CUTOFF = 80           # minimum WRatio (0–100) to keep a fuzzy candidate
WEIGHTS = {"exact": 0.50, "fuzzy": 0.30, "semantic": 0.20}
TOP_K_FINAL = 10            # candidates kept after ranking
LLM_RERANK_THRESHOLD = 0.90 # auto-accept if best final_score >= this, else LLM rerank
TOP_K_RERANK = 5            # candidates returned by the LLM
PARTS_PER_CANDIDATE = 3     # sample parts per candidate shown to the LLM
```

### 8.2 Shared utilities (also imported by item_mapping)
- Normalization **primitives**: `lowercase_text`, `trim_extra_spaces`, `unicode_normalization`,
  `safe_separator_normalization` — exactly as defined in Section 6.3 (None/NaN → `""`).
- `normalize_supplier_name_fuzzy(t)` = fuzzy pipeline; `normalize_supplier_name_semantic(t)` =
  supplier semantic pipeline (Section 6.3).
- `load_master_dataframe() -> DataFrame` — read `MASTER_PARQUET_PATH` (error if missing).
- `load_embedding_model() -> SentenceTransformer` — load from `LOCAL_MODEL_DIR` with
  `trust_remote_code=True`, device `"cuda"` if available else `"cpu"`.
- `_get_bedrock_client()` — lazily create and cache the Bedrock client.
- `load_supplier_resources(seid) -> (faiss_index, metadata_df)` — read the supplier FAISS index
  and metadata for the site; **cache per seid** in a module dict to avoid re-reading.

### 8.3 Candidate generation
A candidate row throughout this module has columns:
`Supplier_Id, Supplier_Name, exact_match, fuzzy_score, cosine_score, matched_via`.

- `prepare_supplier_site_df(df_master, seid) -> DataFrame`
  - Filter `df_master` to `seid`. Keep rows with non-null `Supplier_Id` and non-empty
    `Supplier_Name`. Ensure `supplier_name_fuzzy`/`supplier_name_semantic` are clean strings.
    Reset index.

- `get_exact_supplier_candidates(df_site, query_fuzzy) -> DataFrame`
  - Rows where `supplier_name_fuzzy == query_fuzzy`. For each unique `Supplier_Id` set
    `exact_match=1.0, fuzzy_score=100.0, cosine_score=1.0, matched_via="exact"`. Empty input → empty frame.

- `get_fuzzy_supplier_candidates(df_site, query_fuzzy, top_k=TOP_K_FUZZY, cutoff=FUZZY_CUTOFF) -> DataFrame`
  - Take the **unique** `supplier_name_fuzzy` values; score with
    `rapidfuzz.process.extract(query_fuzzy, choices, scorer=fuzz.WRatio, limit=top_k, score_cutoff=cutoff)`.
    For every matched string, expand back to all its `Supplier_Id`s; set `fuzzy_score=score`,
    `exact_match=0`, `cosine_score=0`, `matched_via="fuzzy"`.

- `get_semantic_supplier_candidates(seid, query_semantic, model, top_k=TOP_K_SEMANTIC) -> DataFrame`
  - `load_supplier_resources(seid)`. Encode the query:
    `model.encode([query_semantic], task="text-matching", convert_to_numpy=True, normalize_embeddings=True).astype(float32)`.
    `scores, indices = index.search(query_vec, min(top_k, index.ntotal))`. Map each returned
    index to its metadata row → `Supplier_Id, Supplier_Name`; set `cosine_score=score`,
    `exact_match=0`, `fuzzy_score=0`, `matched_via="semantic"`.

### 8.4 Backfill (`backfill_candidate_scores(grouped, query_fuzzy, query_semantic, model) -> DataFrame`)
For candidates missing a component score (and not exact matches):
- If `fuzzy_score == 0` and not exact: compute `fuzz.WRatio(query_fuzzy, normalize_supplier_name_fuzzy(candidate.Supplier_Name))`.
- If `cosine_score == 0` and not exact: encode `query_semantic` and the candidates'
  `normalize_supplier_name_semantic(Supplier_Name)` texts (batched), and set `cosine_score` =
  dot product of the (normalized) vectors (`candidate_vecs @ query_vec.T`).

### 8.5 Merge & rank (`merge_and_rank_candidates(exact_df, fuzzy_df, semantic_df, top_k_final=TOP_K_FINAL, query_fuzzy, query_semantic, model) -> DataFrame`)
1. Concatenate the three candidate frames.
2. Group by `Supplier_Id`, taking the **max** of each component score across methods, keeping the
   first `Supplier_Name`, and joining the distinct `matched_via` labels.
3. Run `backfill_candidate_scores`.
4. Compute `fuzzy_score_norm = fuzzy_score / 100` and
   `final_score = 0.50*exact_match + 0.30*fuzzy_score_norm + 0.20*cosine_score`.
5. Sort by `final_score, exact_match, fuzzy_score, cosine_score` (all descending), assign
   `rank = 1..N`, and return the top `top_k_final` rows. (This returned frame is called
   **`ranked_candidates`**.)

### 8.6 LLM rerank
- `get_sample_parts_for_candidate(df_site, supplier_id, n_parts=PARTS_PER_CANDIDATE) -> list[str]`
  — up to `n_parts` distinct `PartName_Descriptive` values for that supplier.
- `rerank_candidates_with_llm(ranked_df, invoice_json, df_site, top_k_rerank, n_parts_per_candidate) -> dict`
  - Build a user message containing: the extracted `supplier_name`; the invoice `line_items`
    (name, qty, rate, amount as text — context only); and, for each candidate in `ranked_df`, a
    line with its `rank, Supplier_Id, Supplier_Name, final_score, fuzzy_score, cosine_score,
    matched_via` plus its sample parts. Ask for the top `top_k_rerank`.
  - Call Bedrock (Section 6.4, supplier rerank variant). Parse the result. For each returned
    `{original_rank, confidence, reason}`, find that rank in `ranked_df` and produce
    `{Supplier_Id, Supplier_Name, final_score, confidence, reason, original_rank}`.
  - Return `{"reranked_candidates": [...], "no_match": bool, "no_match_reason": str}`.

### 8.7 Orchestrator (`map_supplier_name_from_invoice(invoice_json, df_master, model, ...) -> dict`)
1. Read `Site_Id` (int) and `supplier_name` from `invoice_json` (error if `Site_Id` missing).
2. `df_site = prepare_supplier_site_df(df_master, Site_Id)`. If empty → return a result with
   `best_supplier_id=None, best_supplier_name=None, best_score=0.0`.
3. `query_fuzzy = normalize_supplier_name_fuzzy(supplier_name)`;
   `query_semantic = normalize_supplier_name_semantic(supplier_name)`.
4. Generate exact, fuzzy, semantic candidates; `ranked_df = merge_and_rank_candidates(...)`.
5. `best = ranked_df.iloc[0]`. Build the result dict (see below) with `llm_reranked=False`,
   `no_match=False`.
6. **If `best.final_score < LLM_RERANK_THRESHOLD`:** call `rerank_candidates_with_llm`. Set
   `llm_reranked=True`, copy `llm_reranked_candidates`, `no_match`, `no_match_reason`. If the LLM
   returned a usable top candidate (and not `no_match`), overwrite `best_supplier_id/name/score`
   with that candidate. Wrap the LLM call in try/except: on failure, keep the deterministic result.

**Return dict (exact keys — `item_mapping.py` depends on these):**
```
{
  "best_supplier_id":   int | None,
  "best_supplier_name": str | None,
  "best_score":         float,            # 0..1
  "ranked_candidates":  DataFrame,        # the ranked frame from 8.5
  "llm_reranked":       bool,
  "llm_reranked_candidates": list,        # [{Supplier_Id, Supplier_Name, final_score, confidence, reason, original_rank}]
  "no_match":           bool,
  "no_match_reason":    str,
  "query": {"site_id": int, "extracted_supplier": str,
            "supplier_fuzzy_query": str, "supplier_semantic_query": str}
}
```

---

## 9. Component: `item_mapping.py` (BUILD)

Maps each invoice line item to a part of the **already-matched supplier**, by **name only**, then
assembles the final response. Reuse from `supplier_mapping.py`: the normalization primitives,
`load_master_dataframe`, `load_embedding_model`, `_get_bedrock_client`, `BEDROCK_CONFIG`, the path
constants, and `map_supplier_name_from_invoice`.

### 9.1 Config
```
ITEM_WEIGHTS = {"exact": 0.50, "fuzzy": 0.30, "semantic": 0.20}
TOP_K_ITEM_CANDIDATES = 5     # candidate parts kept per line item
SUPPLIER_NOT_FOUND_MSG = "Supplier Name match not found in Database"
ITEM_NOT_FOUND_MSG     = "Item Name match not found in Database"
```
Load the item rerank prompt (`Prompt/item_rerank.yaml`) and schema
(`Response_Schema/item_rerank_schema.json`).

### 9.2 Item normalization (mirror Section 6.3 item pipelines)
- `normalize_item_name_fuzzy(t)` = fuzzy pipeline.
- `normalize_item_name_semantic(t)` = item semantic pipeline (**WITH lowercase**).

### 9.3 Item vector store loader (reuse precomputed vectors via reconstruct)
`load_item_resources(seid) -> (metadata_df, text2row, vectors)`, cached per seid:
- Read `faiss_seid_{seid}_item_name.index` and `metadata_seid_{seid}_item_name.parquet`
  (raise a clear error if either is missing).
- `vectors = index.reconstruct_n(0, index.ntotal).astype(float32)` — all stored vectors, aligned
  to metadata rows.
- `text2row = { str(source_text).strip(): row_index }` for every metadata row — a lookup from the
  normalized item text to its vector row.

### 9.4 Candidates for the matched supplier
- `get_supplier_parts(df_master, seid, supplier_id) -> DataFrame`
  - Filter `df_master` to `seid` and `Supplier_Id == supplier_id`; drop rows with empty
    `PartName_Descriptive`; **deduplicate by `Part_id`**; ensure `item_name_fuzzy` and
    `item_name_semantic` are clean strings. This is the candidate part set.
- `get_candidate_vectors(parts_df, text2row, vectors, model) -> ndarray (C × dim)`
  - For each part, look up its vector by `text2row[item_name_semantic.strip()]`. For any text not
    found (rare), encode just those misses in one batch and fill them in. Stack into one array.

### 9.5 Scoring (name only)
`score_items_against_parts(query_fuzzy, query_vecs, parts_df, cand_vecs) -> (final, exact, fuzzy, cosine)`
where each return is a matrix of shape **(M line items × C candidates)**:
- `fuzzy = rapidfuzz.process.cdist(query_fuzzy, parts_df.item_name_fuzzy, scorer=fuzz.WRatio)` (0–100).
- `exact[i,j] = 1.0 if query_fuzzy[i] == parts_df.item_name_fuzzy[j] else 0.0`.
- `cosine = query_vecs @ cand_vecs.T` (0–1, vectors normalized).
- `final = 0.50*exact + 0.30*(fuzzy/100) + 0.20*cosine`.

`build_item_candidates(parts_df, final, exact, fuzzy, cosine, top_k) -> list[list[dict]]`
- For each line item `i`, take the `top_k` candidates by `final[i]`. Each candidate dict:
  `{rank (1..k), Part_id, part_name (=PartName_Descriptive), exact, fuzzy, cosine, final}`.

### 9.6 Batched LLM rerank (one call per invoice)
`rerank_items_with_llm(uncertain, line_items) -> list[dict]` where `uncertain` is a list of
`(item_index, candidate_list)`:
- Build ONE user message containing, for each uncertain item: its `item_index`, the extracted
  `item_name`, and its candidate parts listed by `rank` with `Part_id`, part name, and the
  component scores. **Do not include price/qty/amount.**
- Call Bedrock (Section 6.4, item rerank variant). Return the parsed `items` list, where each
  element is `{item_index, best_candidate_rank, confidence, no_match, reason}`.

### 9.7 Output shaping helpers
- `_clean(v)` — returns `None` if `v` is `None` or a string whose stripped upper-case is one of
  `"", "NA", "N/A", "NULL", "NONE"`; otherwise returns `v`. (Enforces "missing → null".)
- `_to_int(v)` — coerce numpy/pandas integer ids to a plain `int`, or `None`.
- `_base(it)` — `{"extracted_item_name": _clean(item_name), "qty": _clean(quantity),
  "unit_price": _clean(rate), "line_total_price": _clean(amount)}` from a line item.
- `_matched_item(it, chosen, confidence, reason)` — `_base(it)` plus
  `part_id=_to_int(chosen.Part_id), part_name=chosen.part_name, match_confidence, match_reason`.
- `_not_found_item(it)` — `_base(it)` plus `part_id=None, part_name=None, match_confidence=0,
  match_reason=ITEM_NOT_FOUND_MSG`.
- `_accepted_item(it, best)` — `_matched_item` with `confidence=100` and a reason like
  "Exact name match to catalog part."
- `_resolve_uncertain(it, cands, decision)` — if `decision` is None → `_fallback_item`; if
  `decision.no_match` or `best_candidate_rank` is None/0 → `_not_found_item`; else pick the
  candidate whose `rank == best_candidate_rank` (if not found → `_not_found_item`) and map
  confidence high/medium/low → 90/70/50.
- `_fallback_item(it, cands)` — used if the LLM call errored: take the top candidate
  (`cands[0]`) if any, else `_not_found_item`; confidence = `round(final*100)`.
- `_unmatched_all(line_items)` — `[_not_found_item(it) for it in line_items]`.

### 9.8 Item orchestrator (`map_line_items_from_invoice(invoice_json, supplier_result, df_master, model, top_k_candidates=TOP_K_ITEM_CANDIDATES) -> dict`)
1. `line_items = invoice_json.get("line_items") or []`; `seid = int(Site_Id)`;
   `supplier_id = supplier_result["best_supplier_id"]`.
2. If no line items → `{"supplier_id": supplier_id, "items": []}`.
3. If `supplier_id is None` or `supplier_result.no_match` → `{"supplier_id": supplier_id,
   "items": _unmatched_all(line_items)}`.
4. `parts_df = get_supplier_parts(...)`; if empty → `_unmatched_all(line_items)`.
5. `load_item_resources(seid)`; `cand_vecs = get_candidate_vectors(...)`.
6. Build `query_fuzzy` (per item, fuzzy pipeline) and `query_semantic` (per item, item semantic
   pipeline). Encode **all** `query_semantic` in **one** `model.encode([...], task="text-matching",
   normalize_embeddings=True)` call → `query_vecs`.
7. Compute score matrices; `cands_per_item = build_item_candidates(...)`.
8. **Gate:** for each item, if its top candidate has `exact >= 1.0` → `_accepted_item` (no LLM);
   otherwise add `(i, candidates)` to `uncertain`.
9. If `uncertain`: call `rerank_items_with_llm` once; map each returned decision back by
   `item_index` via `_resolve_uncertain`. On exception, use `_fallback_item` for all uncertain.
10. Return `{"supplier_id", "supplier_name": supplier_result["best_supplier_name"], "items": [...]}`
    (items in original order).

### 9.9 Final response assembly
- `_supplier_confidence(supplier_result) -> int` — if LLM-reranked and has candidates: map the top
  candidate's confidence high/medium/low → 90/70/50; else `round(best_score*100)`.
- `_supplier_reason(supplier_result) -> str` — if LLM-reranked: the top candidate's reason; else a
  generated sentence stating the matched supplier and `best_score`.
- `build_invoice_response(invoice_json, df_master, model, invoice_num=1, pages="Page 1") -> dict`
  - Start with `invoice_obj` = `{invoice_num, pages, vendor_name=_clean(supplier_name),
    supplier_id=None, supplier_name=None, supplier_confidence=0, supplier_match_reason="",
    status="success", error=None, items=[]}`.
  - `supplier_result = map_supplier_name_from_invoice(...)`.
  - **If supplier not found** (`best_supplier_id is None` or `no_match`): set
    `supplier_match_reason = SUPPLIER_NOT_FOUND_MSG`, set `items = _unmatched_all(line_items)`
    (echo the line items as not found), and return — **do not run item matching**.
  - Else set `supplier_id=_to_int(best_supplier_id)`, `supplier_name`, `supplier_confidence`,
    `supplier_match_reason`; then `items = map_line_items_from_invoice(...).items`. Return `invoice_obj`.
- `build_file_response(invoice_json, df_master, model, file_name, total_pages=1) -> dict`
  - Try: `invoice_obj = build_invoice_response(...)`; return
    `{"file": file_name, "status": "success", "total_pages", "error": None, "invoices": [invoice_obj]}`.
  - Except: return `{"file": file_name, "status": "error", "total_pages", "error": str(e), "invoices": []}`.

---

## 10. Component: `main.py` (BUILD) — single entry point

- Constants: `REQUEST_FILE = request_structure.json`; `INVOICE_PATH` = a local invoice file
  (S3 download skipped). Module-level caches `_DF_MASTER`, `_MODEL`.
- `_get_resources()` — load `df_master` and the model once (cache), return both.
- `load_request(x)` — accept a dict, a JSON string, or a path to a JSON file; return the dict.
  The request file shape is `{"siteid": int, "signedUrl": str}` (you use only `siteid`).
- `process_invoice(invoice_path, site_id, password=None) -> dict`
  1. `df_master, model = _get_resources()`.
  2. `extracted = extract_invoice(str(invoice_path), password)`  → `{file, total_pages, entities}`.
  3. `invoice_json = {**extracted["entities"], "Site_Id": site_id}`.
  4. `return build_file_response(invoice_json, df_master, model, file_name=extracted["file"],
     total_pages=extracted["total_pages"])`.
  5. Wrap in try/except → on failure return a `status="error"` response.
- `__main__`: `req = load_request(REQUEST_FILE)`; `response = process_invoice(INVOICE_PATH,
  req["siteid"])`; print it and write `final_response.json`.

---

## 11. Scoring & Decision Logic (summary)

- Weights are `exact 0.50 / fuzzy 0.30 / semantic 0.20` for both stages.
- **The score distribution is bimodal:** an exact match scores **1.0**; any non-exact match can
  reach at most **0.5** (because `exact_match=0` removes the 0.50 term). Therefore any threshold
  strictly between 0.5 and 1.0 means "auto-accept exact matches, send everything else to the LLM."
- **Supplier:** `LLM_RERANK_THRESHOLD = 0.90` (auto-accepts exact; reranks the rest).
- **Item:** explicit rule "auto-accept only when the top candidate's `exact == 1.0`; **all**
  non-exact items go to the single batched LLM rerank."
- **Rationale:** the LLM is a disambiguation tool — valuable when deterministic scores are weak,
  risky/wasteful on certain exact matches. Item matching is more ambiguous and its rerank is
  batched into one call per invoice, so routing all non-exact items to the LLM is cost-bounded.
- **Accuracy can only be proven with a labeled test set.** Build one (≥30–50 invoices with known
  `Supplier_Id`/`Part_id`) to tune thresholds and validate.

---

## 12. Output Format (final response — the system's contract)

```json
{
  "file": "02429994.TIF",
  "status": "success",
  "total_pages": 1,
  "error": null,
  "invoices": [
    {
      "invoice_num": 1,
      "pages": "Page 1",
      "vendor_name": "<extracted supplier name (null if absent)>",
      "supplier_id": 224491,
      "supplier_name": "<matched Supplier_Name (null if not found)>",
      "supplier_confidence": 99,
      "supplier_match_reason": "<reason or 'Supplier Name match not found in Database'>",
      "status": "success",
      "error": null,
      "items": [
        {
          "extracted_item_name": "<extracted item name (null if absent)>",
          "qty": null,
          "unit_price": "<extracted rate (null if absent)>",
          "line_total_price": "<extracted amount (null if absent)>",
          "part_id": 50446,
          "part_name": "<matched PartName_Descriptive (null if not found)>",
          "match_confidence": 72,
          "match_reason": "<reason or 'Item Name match not found in Database'>"
        }
      ]
    }
  ]
}
```

Field rules:
- `extracted_item_name`, `qty`, `unit_price`, `line_total_price` are **copied** from extraction
  (null if absent); they never affect matching.
- `supplier_confidence` / `match_confidence`: exact auto-accept → 100; LLM high/medium/low →
  90/70/50; not found → 0; supplier non-rerank → `round(best_score*100)`.
- When supplier is not found: `supplier_id`/`supplier_name` are null, `supplier_match_reason` =
  `"Supplier Name match not found in Database"`, and items are echoed as not-found.

---

## 13. Configuration Reference

| Setting | Where | Value |
|---|---|---|
| LLM model / temp / tokens | `bedrock_config.yaml` (provided) | Claude Haiku 4.5 ARN, `temperature=0`, `max_tokens=4096`, `timeout=300`, `retry_delay=10` |
| AWS profile / region | code | `shrey_bedrock` / `us-east-1` |
| Embedding model dir | code | `./models/jina-embeddings-v5-text-small-text-matching` |
| Image max edge / PDF zoom | `extract_entities.py` | 1568 px / 2.5× |
| Supplier knobs | `supplier_mapping.py` | `TOP_K_FUZZY=10, TOP_K_SEMANTIC=10, FUZZY_CUTOFF=80, TOP_K_FINAL=10, LLM_RERANK_THRESHOLD=0.90, TOP_K_RERANK=5, PARTS_PER_CANDIDATE=3` |
| Item knobs | `item_mapping.py` | `ITEM_WEIGHTS=0.5/0.3/0.2, TOP_K_ITEM_CANDIDATES=5` |

---

## 14. Risks, Pitfalls & Mitigations

| # | Issue | Impact | Mitigation |
|---|---|---|---|
| 1 | Re-encoding candidate embeddings at query time | High latency | Reuse precomputed vectors via FAISS `reconstruct_n` + `source_text→row` lookup; encode only the **query** (once, batched) |
| 2 | FAISS `search` returns whole-site top-k; cannot restrict to one supplier | Wrong/missed item candidates | For items, reconstruct vectors and filter to the supplier's parts; do not use FAISS `search` for item scoping |
| 3 | Item semantic normalizer **lowercases**, supplier's does not | Wrong cosine scores | Implement the two semantic pipelines separately, exactly per Section 6.3 |
| 4 | Modules read `bedrock_config.yaml` / data via relative paths | `FileNotFoundError` | **Run from the project root** (or make paths `BASE_DIR`-relative) |
| 5 | Bedrock client built at import time | Import fails without AWS creds | `aws sso login` before running; or lazy-init the client |
| 6 | `max_tokens=4096` with all non-exact items batched | Truncated/invalid JSON for large invoices | Watch response size; raise `max_tokens`, cap `TOP_K_ITEM_CANDIDATES`, or chunk the batch |
| 7 | Item metadata lacks `Supplier_Id` | Cannot filter the index directly | Get candidates from `master_dataset` (has `Supplier_Id`); get their vectors by `source_text` lookup (embedding is a pure function of text) |
| 8 | `source_text` lookup miss | Missing candidate vector | Fallback: encode just the missing candidate texts |
| 9 | numpy/pandas `Int64` ids aren't JSON-native | Serialization errors | `_to_int()`; also dump with `json.dumps(..., default=str)` |
| 10 | Treating missing values as defaults | Wrong data | Enforce null via `_clean()` (maps `""`,`"NA"`,`"N/A"`,`"NULL"`,`"NONE"`→ null) |
| 11 | Per-site **item** FAISS index missing | `FileNotFoundError` | Ensure `precompute_embeddings.py` has been run; raise a clear error |
| 12 | Reloading model/parquet per call | Slow | Cache: `_get_resources()` and the per-seid resource caches |
| 13 | LLM wraps JSON in ```` ```json ```` fences | Parse error | Strip fences before `json.loads` (Section 6.4) |
| 14 | LLM rerank call fails (network/throttle) | Pipeline break | try/except: supplier keeps deterministic ranking; items fall back to top name-scored candidate |
| 15 | Wrong top-1 supplier dooms all its items | Cascade failure | Supplier rerank already uses item names as a signal; acceptable for v1 (future: top-k supplier fallback) |
| 16 | Embeddings must be normalized for cosine=dot | Wrong scores | Always `normalize_embeddings=True`; indexes are `IndexFlatIP` |
| 17 | Duplicate part rows per supplier | Skewed candidates | `get_supplier_parts` dedups by `Part_id` |

---

## 15. Non-Functional Requirements
- **Determinism:** `temperature=0` for extraction and all reranking.
- **Latency:** exactly one query-encode per stage per invoice; LLM calls minimized (exact-only
  auto-accept; a single batched item rerank).
- **Cost:** LLM invoked only on ambiguous cases.
- **Scalability:** per-site indexes keep search spaces small; heavy resources cached.
- **Robustness:** graceful degradation on LLM/IO failures; null-safe output; never crash the run.
- **Maintainability:** shared utilities live in `supplier_mapping.py` and are imported by
  `item_mapping.py` (no duplication).

---

## 16. Setup & Run

**One-time data prep (provided scripts):**
1. Install dependencies (Section 5). 2. Configure `.env` for the DB.
3. `python export_parquet.py` → `data/master_dataset.parquet`.
4. `python precompute_embeddings.py` → downloads the embedding model (first run) and builds
   per-site FAISS indexes + metadata for **both** `supplier_name` and `item_name`.

**Per run:**
```bash
aws sso login --profile shrey_bedrock
python main.py        # run from the project root
```
Prints the JSON and writes `final_response.json`. Set `INVOICE_PATH` to the target invoice.

---

## 17. Verification & Testing
- **Syntax:** `python -m py_compile main.py item_mapping.py supplier_mapping.py extract_entities.py`.
- **Happy path:** run `main.py` on a known invoice; confirm the Section 12 structure; verify exact
  items auto-accept (no LLM), non-exact items go through one batched rerank, and unrelated charges
  (tax/freight) return `part_id=null`.
- **Not-found paths:** an invoice whose supplier is absent (expect supplier not-found message and
  echoed not-found items); an invoice with non-catalog line items (expect per-item not-found).
- **Accuracy:** evaluate top-1 accuracy on a labeled set; tune thresholds.

---

## 18. Future Enhancements
- Wire S3 download from `signedUrl` (stdlib `urllib`); add a local-path toggle.
- Support multiple invoices per file (the `invoices[]` array already allows it).
- Confidence-gated **top-k supplier** fallback (let item matches validate the supplier).
- Add `Supplier_Id` to item metadata to filter the FAISS index directly (remove the text lookup).
- Optional price/quantity validation as a **post-match** sanity check (kept out of matching).
- Structured logging; retry/backoff for Bedrock throttling; batch/CLI runner.
```
