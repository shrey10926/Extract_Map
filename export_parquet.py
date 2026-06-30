import os, pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from pathlib import Path

from text_normalization import (
    normalize_supplier_name_fuzzy,
    normalize_supplier_name_semantic,
    normalize_item_name_fuzzy,
    normalize_item_name_semantic,
)


# =========================================================
# CONFIG
# =========================================================

OUTPUT_DIR = Path(r"./data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_FILE = OUTPUT_DIR / "master_dataset.parquet"

load_dotenv()

server = os.getenv("DB_HOST")
database = os.getenv("DB_NAME")
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
driver = os.getenv("DB_DRIVER")
trust = os.getenv("DB_TRUST_CERT", "yes")

odbc_str = (
    f"DRIVER={{{driver}}};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    f"TrustServerCertificate={trust};"
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

    -- =========================================
    -- SUPPLIER INFORMATION
    -- =========================================
    s.Supplier_Id,
    s.Supplier_Name,

    -- =========================================
    -- PRODUCT INFORMATION
    -- =========================================
    p.Part_id,
    p.PartName_Descriptive,

    -- =========================================
    -- SUPPLIER PRODUCT MAPPING
    -- =========================================
    sp.Supplier_Part_Id,
    sp.Sup_Part_Code,
    sp.Reg_Price

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
    "Sup_Part_Code"
]

for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

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