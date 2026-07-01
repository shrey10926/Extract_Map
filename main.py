from __future__ import annotations
import json, time, logging
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.request import urlopen
from urllib.error import HTTPError

from extract_entities_3 import extract_invoice, ExtractionValidationError
from supplier_mapping_3 import load_master_dataframe, load_embedding_model
from item_mapping_3 import build_file_response

BASE_DIR = Path(__file__).resolve().parent
REQUEST_FILE = BASE_DIR / "sample_request1.json"
DOWNLOAD_TIMEOUT = 60   # seconds, for the invoice download
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024   # 25 MB cap on the downloaded invoice

# URL scheme allowlist. Relaxed (None) for the current public-URL test phase.
# When the client switches to pre-signed S3 URLs, set this to {"https"} (and optionally
# add an S3 host check) to block file:// / internal-host SSRF.
ALLOWED_URL_SCHEMES = None

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
    """Download the invoice from a (pre-signed) URL into memory — one GET, no temp file.

    Guards: optional scheme allowlist, explicit timeout, HTTP status check, a hard size
    cap (defends against a missing/lying Content-Length), and an empty-body check.
    Returns (data_bytes, filename, content_type).
    """
    parsed = urlparse(signed_url)
    if ALLOWED_URL_SCHEMES is not None and parsed.scheme.lower() not in ALLOWED_URL_SCHEMES:
        raise ValueError(f"URL scheme '{parsed.scheme}' is not allowed.")

    try:
        resp = urlopen(signed_url, timeout=timeout)
    except HTTPError as e:
        # Expired / forbidden / missing pre-signed URL -> clear, safe message.
        raise ValueError(f"Failed to download invoice: HTTP {e.code}.") from None

    with resp:
        status = getattr(resp, "status", None)
        if status is not None and not (200 <= status < 300):
            raise ValueError(f"Failed to download invoice: HTTP {status}.")

        clen = resp.headers.get("Content-Length")
        if clen and clen.isdigit() and int(clen) > MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"Invoice exceeds the {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB size limit."
            )

        # Read one byte past the cap so we can detect overflow even without a header.
        data = resp.read(MAX_DOWNLOAD_BYTES + 1)
        if len(data) > MAX_DOWNLOAD_BYTES:
            raise ValueError(
                f"Invoice exceeds the {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB size limit."
            )
        content_type = resp.headers.get("Content-Type")

    if not data:
        raise ValueError("Downloaded invoice is empty.")

    return data, _filename_from_url(signed_url), content_type


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
        data, file_name, content_type = download_invoice_bytes(signed_url)

        # 2) Extract entities (opened from the in-memory bytes) -> {file, total_pages, entities}
        extracted = extract_invoice(data, filename=file_name, password=password,
                                    content_type=content_type)

        # 3) Attach Site_Id (mapping is scoped by site)
        invoice_json = {**extracted["entities"], "Site_Id": site_id}

        # 4) Supplier + item mapping -> final structured response
        return build_file_response(
            invoice_json, df_master, model,
            file_name=extracted["file"],
            total_pages=extracted["total_pages"],
        )

    except ExtractionValidationError as e:
        # Extraction produced nothing usable -> fail early with a clear, safe message.
        return {"file": file_name, "status": "invalid_extraction", "total_pages": 0,
                "error": str(e), "invoices": []}
    except ValueError as e:
        # Deliberate guard failures (bad URL scheme, oversized/empty download,
        # unsupported file type, unrenderable doc). Messages here are safe to surface.
        return {"file": file_name, "status": "error", "total_pages": 0,
                "error": str(e), "invoices": []}
    except Exception:
        # Unexpected internal error: log full detail server-side, return a generic message.
        logging.exception("Unexpected error processing invoice %s", file_name)
        return {"file": file_name, "status": "error", "total_pages": 0,
                "error": "Failed to process the invoice due to an internal error.",
                "invoices": []}


if __name__ == "__main__":
    start = time.time()
    response = process_invoice(REQUEST_FILE)

    print(json.dumps(response, indent=2, default=str))
    with open(BASE_DIR / "final_response.json", "w", encoding="utf-8") as f:
        json.dump(response, f, indent=4, ensure_ascii=False, default=str)
    
    print(f"Total time taken is: {time.time() - start} seconds!")