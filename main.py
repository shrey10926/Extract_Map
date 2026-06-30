from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import urlopen

from extract_entities import extract_invoice
from supplier_mapping import load_master_dataframe, load_embedding_model
from item_mapping import build_file_response

BASE_DIR = Path(__file__).resolve().parent
REQUEST_FILE = BASE_DIR / "sample_request2.json"
DOWNLOAD_TIMEOUT = 60   # seconds, for the invoice download

# Heavy resources are loaded once and reused across invoices.
_DF_MASTER = None
_MODEL = None


def _get_resources():
    global _DF_MASTER, _MODEL
    if _DF_MASTER is None:
        _DF_MASTER = load_master_dataframe()
    if _MODEL is None:
        _MODEL = load_embedding_model()
    return _DF_MASTER, _MODEL


def load_request(request_input):
    """Accept a dict, a JSON string, or a path to a request JSON ({siteid, signedUrl})."""
    if isinstance(request_input, dict):
        return request_input
    s = str(request_input).strip()
    if Path(s).exists():
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(s)


def _filename_from_url(signed_url: str) -> str:
    return Path(unquote(urlparse(signed_url).path)).name or "invoice"


def download_invoice_bytes(signed_url: str, timeout: int = DOWNLOAD_TIMEOUT):
    """Download the invoice from a (pre-signed) URL into memory — one GET, no temp file."""
    with urlopen(signed_url, timeout=timeout) as resp:
        return resp.read(), _filename_from_url(signed_url)


def process_invoice(request, password: str | None = None) -> dict:
    """
    End-to-end: read request -> download invoice (in memory) -> extract -> supplier mapping
    -> item mapping -> final response.

    `request` is a dict / JSON string / path to a request JSON ({siteid, signedUrl}).
    Returns the final {file, status, total_pages, error, invoices:[...]} structure.
    """
    req = load_request(request)
    site_id = req.get("siteid")
    signed_url = req.get("signedUrl")
    file_name = _filename_from_url(signed_url) if signed_url else None

    try:
        df_master, model = _get_resources()

        # 1) Download the invoice from the pre-signed S3 URL into memory (no temp file)
        data, file_name = download_invoice_bytes(signed_url)

        # 2) Extract entities (opened from the in-memory bytes) -> {file, total_pages, entities}
        extracted = extract_invoice(data, filename=file_name, password=password)

        # 3) Attach Site_Id (mapping is scoped by site)
        invoice_json = {**extracted["entities"], "Site_Id": site_id}

        # 4) Supplier + item mapping -> final structured response
        return build_file_response(
            invoice_json, df_master, model,
            file_name=extracted["file"],
            total_pages=extracted["total_pages"],
        )

    except Exception as e:
        return {"file": file_name, "status": "error", "total_pages": 0,
                "error": str(e), "invoices": []}


if __name__ == "__main__":
    response = process_invoice(REQUEST_FILE)

    print(json.dumps(response, indent=2, default=str))
    with open(BASE_DIR / "final_response.json", "w", encoding="utf-8") as f:
        json.dump(response, f, indent=4, ensure_ascii=False, default=str)
