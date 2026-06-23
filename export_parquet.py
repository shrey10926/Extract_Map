import unicodedata, re, os, pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from pathlib import Path
from datetime import datetime


# =========================================================
# CONFIG
# =========================================================

SUPPLIER_STOP_WORDS = {
    "llc",
    "ltd",
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
    "pvt",
    "plc",
    "llp"
}


OUTPUT_DIR = Path(r"./data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_FILE = OUTPUT_DIR / "master_dataset.parquet"

load_dotenv()

server = os.getenv("DB_HOST")
database = os.getenv("DB_NAME")
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")

odbc_str = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    f"TrustServerCertificate=yes;"
)

connect_url = (
    f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_str)}"
)

engine = create_engine(connect_url)


# =========================================================
# MASTER JOIN QUERY
# =========================================================
#
# This query:
# 1. Joins all relevant tables
# 2. Removes unnecessary columns
# 3. Keeps only columns required for AI matching
# 4. Filters inactive/deleted rows
#
# =========================================================

query = """
SELECT

    -- =========================================
    -- SITE INFORMATION
    -- =========================================
    p.seid,

    e.entity_name,
    e.entity_shortname,

    -- =========================================
    -- SUPPLIER INFORMATION
    -- =========================================
    s.Supplier_Id,
    s.Supplier_Name,
    s.VendorId,
    s.EmailAddress,
    s.CountryName,

    -- =========================================
    -- PRODUCT INFORMATION
    -- =========================================
    p.Part_id,

    p.PartName_Descriptive,
    p.PartFull_Description,

    p.GTIN,
    p.GTIN_UPC,

    -- =========================================
    -- SUPPLIER PRODUCT MAPPING
    -- =========================================
    sp.Supplier_Part_Id,

    sp.Sup_Part_Code,

    sp.Reg_Price,
    sp.Min_Qty

FROM Product.Part p

INNER JOIN SupOrder.Supplier_Parts sp
    ON p.Part_id = sp.Part_Id

INNER JOIN SupOrder.Supplier s
    ON sp.Supplier_Id = s.Supplier_Id

LEFT JOIN Sites.entity e
    ON p.seid = e.seid

WHERE

    -- =========================================
    -- FILTER DELETED / INACTIVE DATA
    -- =========================================

    ISNULL(s.DeleteFlag, 0) = 0
    AND ISNULL(sp.InActive, 0) = 0
"""

# =========================================================
# LOAD DATA
# =========================================================

print("Executing SQL query...")

df = pd.read_sql(query, engine)

print(f"Rows fetched: {len(df)}")
print(f"Columns fetched: {len(df.columns)}")

print(f"data types before optimization:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")


# =========================================================
# OPTIONAL CLEANING
# =========================================================

print("Cleaning data...")
# df = pd.read_excel(r"db_exports/combined_raw_data.xlsx")

# Strip whitespace
object_cols = df.select_dtypes(include=["object"]).columns
string_cols = df.select_dtypes(include=["str"]).columns

for col in object_cols:
    df[col] = df[col].astype(str).str.strip()

for col in string_cols:
    df[col] = df[col].astype(str).str.strip()

# # Remove duplicate rows
# df.drop_duplicates(inplace=True)

# # Fill NaN text columns
text_cols = [
    "Supplier_Name",
    "PartName_Descriptive",
    "PartFull_Description",
    "Sup_Part_Code"
]

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

# =========================================================
# CREATE SEARCHABLE COMBINED TEXT
# =========================================================
#
# This field will later be used for embeddings
#
# =========================================================

print("Creating combined_text column...")

df["combined_text"] = (
    "Part Name: " + df["PartName_Descriptive"].astype(str)
    # + " | Description: " + df["PartFull_Description"].astype(str)
    + " | Supplier Code: " + df["Sup_Part_Code"].astype(str)
)

# =========================================================
# DATA TYPE OPTIMIZATION
# =========================================================

print("Optimizing datatypes...")

int_cols = [
    "seid",
    "Supplier_Id",
    "Part_id",
    "Supplier_Part_Id"
]

for col in int_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        ).astype("Int64")



# =========================================================
# CORE STRING-BASED NORMALIZATION FUNCTIONS
# =========================================================

def lowercase_text(text: str) -> str:
    """
    Convert text to lowercase.
    """
    if pd.isna(text):
        return ""

    return str(text).lower()


def trim_extra_spaces(text: str) -> str:
    """
    Remove leading/trailing spaces and multiple spaces.
    """
    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def unicode_normalization(text: str) -> str:
    """
    Normalize unicode characters.

    Example:
        Café -> Cafe
    """
    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKD", text)

    text = text.encode("ascii", "ignore").decode("utf-8")

    return text


def remove_relevant_stop_words(
    text: str,
    stop_words: set = SUPPLIER_STOP_WORDS
) -> str:
    """
    Remove supplier suffix words such as:
    LLC, LTD, INC, CORP etc.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    tokens = text.split()

    tokens = [token for token in tokens if token not in stop_words]

    return " ".join(tokens)


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
    # bulbs.com -> bulbs com
    # 3.5oz     -> preserved
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# PIPELINE FUNCTIONS (STRING BASED)
# ========================================================= 

def normalize_supplier_name_fuzzy(text: str) -> str:
    """
    supplier_name_fuzzy pipeline
    """

    text = lowercase_text(text)
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    # text = remove_relevant_stop_words(text)
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


def normalize_item_name_fuzzy(text: str) -> str:
    """
    item_name_fuzzy pipeline
    """

    text = lowercase_text(text)
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    text = safe_separator_normalization(text)
    text = trim_extra_spaces(text)

    return text


def normalize_item_name_semantic(text: str) -> str:
    """
    item_name_semantic pipeline
    """

    text = lowercase_text(text)
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    text = trim_extra_spaces(text)

    return text





# =========================================================
# CREATE NORMALIZED COLUMNS
# =========================================================

df["supplier_name_fuzzy"] = df["Supplier_Name"].apply(
    normalize_supplier_name_fuzzy
)

df["supplier_name_semantic"] = df["Supplier_Name"].apply(
    normalize_supplier_name_semantic
)

df["item_name_fuzzy"] = df["PartName_Descriptive"].apply(
    normalize_item_name_fuzzy
)

df["item_name_semantic"] = df["PartName_Descriptive"].apply(
    normalize_item_name_semantic
)


# # =========================================================
# # SAVE AS PARQUET
# # =========================================================

print("Saving parquet file...")
df.to_excel(r"data/combined_raw_data.xlsx", index=False)
df.to_parquet(
    PARQUET_FILE,
    engine="pyarrow",
    index=False
)

print(f"\nParquet saved successfully:")
print(PARQUET_FILE)

# # =========================================================
# # OPTIONAL SUMMARY
# # =========================================================

print("\n========== DATASET SUMMARY ==========")

print(f"Total Rows      : {len(df)}")
print(f"Total Columns   : {len(df.columns)}")

print("\nUnique Sites:")
print(df["seid"].nunique())

print("\nUnique Suppliers:")
print(df["Supplier_Id"].nunique())

print("\nUnique Products:")
print(df["Part_id"].nunique())
df.columns
# # =========================================================
# # CLOSE CONNECTION
# # =========================================================

# conn.close()

# print("\nMSSQL connection closed.")






# EXPORT ALL DATA AS CSV
# tables = [
#     "SupOrder.Supplier_Parts",
#     "Product.Part",
#     "SupOrder.Supplier",
#     "Sites.entity"
# ]

# os.makedirs("db_exports", exist_ok=True)
# chunksize = 10000

# for table in tables:

#     query = f"SELECT * FROM {table}"

#     chunk_iter = pd.read_sql(
#         query,
#         engine,
#         chunksize=chunksize
#     )
#     output_path = f"db_exports/{table}.csv"

#     first_chunk = True
#     for chunk in chunk_iter:
#         chunk.to_csv(
#             output_path,
#             mode="w" if first_chunk else "a",
#             header=first_chunk,
#             index=False
#         )
#         first_chunk = False

#     print(f"Exported {table}")