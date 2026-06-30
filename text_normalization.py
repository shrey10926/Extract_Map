"""
Single source of truth for text normalization.

These functions are used in TWO places that MUST stay identical:
  - build time   (export_parquet.py)  -> the *_fuzzy / *_semantic parquet columns
  - query time   (supplier_mapping.py, item_mapping.py) -> the extracted name

If the build-time and query-time normalization ever diverge, exact/fuzzy/semantic
matching silently breaks. Keeping them here guarantees they cannot drift.
"""
from __future__ import annotations
import re
import unicodedata
import pandas as pd


# =========================================================
# CORE STRING-BASED NORMALIZATION PRIMITIVES
# =========================================================

def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""
    if pd.isna(text):
        return ""
    return str(text).lower()


def trim_extra_spaces(text: str) -> str:
    """Remove leading/trailing spaces and collapse internal whitespace."""
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


# =========================================================
# PIPELINES (STRING BASED)
# =========================================================

def normalize_supplier_name_fuzzy(text: str) -> str:
    """supplier_name_fuzzy pipeline"""
    text = lowercase_text(text)
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    text = safe_separator_normalization(text)
    text = trim_extra_spaces(text)
    return text


def normalize_supplier_name_semantic(text: str) -> str:
    """supplier_name_semantic pipeline (case preserved for the embedding model)"""
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    text = trim_extra_spaces(text)
    return text


def normalize_item_name_fuzzy(text: str) -> str:
    """item_name_fuzzy pipeline"""
    text = lowercase_text(text)
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    text = safe_separator_normalization(text)
    text = trim_extra_spaces(text)
    return text


def normalize_item_name_semantic(text: str) -> str:
    """item_name_semantic pipeline (items are lowercased; suppliers are not)"""
    text = lowercase_text(text)
    text = trim_extra_spaces(text)
    text = unicode_normalization(text)
    text = trim_extra_spaces(text)
    return text
