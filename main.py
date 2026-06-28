# main.py — end-to-end invoice extraction + mapping pipeline
# Run from the project root:  python main.py
# Requires:  aws sso login --profile shrey_bedrock
#
# NOTE: S3 download from signedUrl is intentionally skipped for now.
#       Point INVOICE_PATH at a local invoice file. To re-enable downloading
#       later, fetch the file from req["signedUrl"] and pass that path instead.
from __future__ import annotations
import json
from pathlib import Path

from extract_entities import extract_invoice
from supplier_mapping import load_master_dataframe, load_embedding_model
from item_mapping import build_file_response

BASE_DIR = Path(__file__).resolve().parent
REQUEST_FILE = BASE_DIR / "request_structure.json"

# S3 download skipped — use a local invoice file.
INVOICE_PATH = BASE_DIR / "1008_1234.pdf"

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


def process_invoice(invoice_path, site_id, password: str | None = None) -> dict:
    """
    End-to-end (local file): extract -> supplier mapping -> item mapping -> final response.

    S3 download is skipped; pass a local invoice file path.
    Returns the final {file, status, total_pages, error, invoices:[...]} structure.
    """
    file_name = Path(invoice_path).name
    try:
        df_master, model = _get_resources()

        # 1) Extract entities  ->  {file, total_pages, entities}
        extracted = extract_invoice(str(invoice_path), password=password)

        # 2) Attach Site_Id (mapping is scoped by site)
        invoice_json = {**extracted["entities"], "Site_Id": site_id}

        # 3) Supplier + item mapping -> final structured response
        return build_file_response(
            invoice_json, df_master, model,
            file_name=extracted["file"],
            total_pages=extracted["total_pages"],
        )

    except Exception as e:
        return {"file": file_name, "status": "error", "total_pages": 0,
                "error": str(e), "invoices": []}


if __name__ == "__main__":
    req = load_request(REQUEST_FILE)
    site_id = req.get("siteid")

    response = process_invoice(INVOICE_PATH, site_id)

    print(json.dumps(response, indent=2, default=str))
    with open(BASE_DIR / "final_response.json", "w", encoding="utf-8") as f:
        json.dump(response, f, indent=4, ensure_ascii=False, default=str)
