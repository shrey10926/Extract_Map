# aws sso login --profile shrey_bedrock
from pathlib import Path
import io, time, json, yaml, boto3, fitz
from typing import Optional, List, Dict, Any
from PIL import Image


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = Path("./saved_images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CONFIG_FILE = BASE_DIR / "bedrock_config.yaml"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    APP_CONFIG = yaml.safe_load(f)


PROMPT_FILE = BASE_DIR / "Prompt" / "prompt.yaml"
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt = yaml.safe_load(f)
SYSTEM_PROMPT = prompt["system_prompt"]


SCHEMA_FILE = BASE_DIR / "Response_Schema" / "response_schema_updated.json"
with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
    response_schema = json.load(f)

session = boto3.Session(profile_name="shrey_bedrock")

client = session.client(
    "bedrock-runtime",
    region_name="us-east-1"
)


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def preprocess_image(
    img: Image.Image,
    output_path: str = None,
    max_edge: int = 1568
) -> bytes:
    """
    Convert image to RGB, resize if required,
    return PNG bytes and optionally save.
    """

    img = img.convert("RGB")
    width, height = img.size
    if max(width, height) > max_edge:
        if width > height:
            new_width = max_edge
            new_height = int(height * max_edge / width)
        else:
            new_height = max_edge
            new_width = int(width * max_edge / height)

        img = img.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="PNG",
        optimize=True
    )

    png_bytes = buffer.getvalue()

    # Save debug image
    if output_path:
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(png_bytes)

    return png_bytes

# =============================================================================
# PDF PROCESSING
# =============================================================================

def process_pdf(
    data: bytes,
    password: Optional[str] = None,
    base_name: str = "invoice",
    save_images: bool = False
) -> List[Dict[str, Any]]:

    processed_pages = []

    doc = fitz.open(stream=data, filetype="pdf")   # open from memory, no temp file

    try:

        if doc.is_encrypted:

            if not password:
                raise ValueError("PDF is encrypted but no password was provided.")

            if not doc.authenticate(password):
                raise ValueError("Incorrect password for the PDF.")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            matrix = fitz.Matrix(2.5, 2.5)
            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False
            )

            img = Image.frombytes(
                "RGB",
                [pix.width, pix.height],
                pix.samples
            )

            save_path = None
            if save_images:
                output_dir = OUTPUT_DIR / base_name
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(output_dir / f"page_{page_num + 1}.png")

            png_bytes = preprocess_image(
                img=img,
                output_path=save_path
            )

            processed_pages.append(
                {
                    "page_number": page_num + 1,
                    "png_bytes": png_bytes,
                    "saved_path": save_path
                }
            )

    finally:
        doc.close()

    return processed_pages


# =============================================================================
# TIFF PROCESSING
# =============================================================================

def process_tiff(
    data: bytes,
    base_name: str = "invoice",
    save_images: bool = False
) -> List[Dict[str, Any]]:

    processed_pages = []
    img = Image.open(io.BytesIO(data))   # open from memory

    page_num = 1
    while True:

        try:
            frame = img.copy()

            save_path = None
            if save_images:
                output_dir = OUTPUT_DIR / base_name
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = str(output_dir / f"page_{page_num}.png")

            print(f"Processing page: {page_num}")
            png_bytes = preprocess_image(
                img=frame,
                output_path=save_path
            )

            processed_pages.append(
                {
                    "page_number": page_num,
                    "png_bytes": png_bytes,
                    "saved_path": save_path
                }
            )

            page_num += 1
            img.seek(img.tell() + 1)

        except EOFError:
            break

    return processed_pages


# =============================================================================
# JPG / PNG / WEBP / BMP
# =============================================================================

def process_image_file(
    data: bytes,
    base_name: str = "invoice",
    save_images: bool = False
) -> List[Dict[str, Any]]:

    img = Image.open(io.BytesIO(data))   # open from memory

    save_path = None
    if save_images:
        output_dir = OUTPUT_DIR / base_name
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(output_dir / f"{base_name}.png")

    png_bytes = preprocess_image(
        img=img,
        output_path=save_path
    )

    return [
        {
            "page_number": 1,
            "png_bytes": png_bytes,
            "saved_path": save_path
        }
    ]


# =============================================================================
# DOCUMENT ROUTER
# =============================================================================

def _load_source(
    source,
    filename: Optional[str] = None
):
    """
    Return (data_bytes, name, ext) from either:
      - raw bytes (then `filename` with an extension is required), or
      - a filesystem path (str / Path).
    """
    if isinstance(source, (bytes, bytearray)):
        if not filename:
            raise ValueError("filename (with extension) is required when passing raw bytes.")
        name = Path(filename).name
        return bytes(source), name, Path(name).suffix.lower()

    p = Path(source)
    return p.read_bytes(), p.name, p.suffix.lower()


def convert_document(
    data: bytes,
    ext: str,
    password: Optional[str] = None,
    base_name: str = "invoice",
    save_images: bool = False
) -> List[Dict[str, Any]]:

    ext = ext.lower()

    if ext == ".pdf":
        return process_pdf(data, password=password, base_name=base_name, save_images=save_images)

    elif ext in (".tif", ".tiff"):
        return process_tiff(data, base_name=base_name, save_images=save_images)

    elif ext in (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    ):
        return process_image_file(data, base_name=base_name, save_images=save_images)

    raise ValueError(
        f"Unsupported file type: {ext}"
    )


# =============================================================================
# BUILD BEDROCK CONTENT BLOCKS
# =============================================================================

def build_content_blocks(
    pages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    content_blocks = []

    for page in pages:

        content_blocks.append(
            {
                "image": {
                    "format": "png",
                    "source": {
                        "bytes": page["png_bytes"]
                    }
                }
            }
        )

    return content_blocks


# =============================================================================
# CLAUDE EXTRACTION
# =============================================================================

def extract_invoice(
    source,
    password: Optional[str] = None,
    filename: Optional[str] = None,
    save_images: bool = True
) -> dict:
    """
    `source` may be raw bytes (then `filename` with an extension is required) or a file path.
    Set `save_images=True` to also write the preprocessed page PNGs to disk (off by default
    to minimize IO).

    Returns:
        {
            "file": <source filename>,
            "total_pages": <number of pages processed>,
            "entities": <extracted entities dict>
        }
    """

    data, name, ext = _load_source(source, filename)

    pages = convert_document(
        data=data,
        ext=ext,
        password=password,
        base_name=Path(name).stem,
        save_images=save_images
    )

    print(f"Pages processed: {len(pages)}")

    content_blocks = build_content_blocks(pages)

    response = client.converse(
        modelId=APP_CONFIG["api"]["model_name"],
        system=[
            {
                "text": SYSTEM_PROMPT
            }
        ],
        messages=[
            {
                "role": "user",
                "content": content_blocks
            }
        ],
        outputConfig={
            "textFormat": {
                "type": "json_schema",
                "structure": {
                    "jsonSchema": {
                        "name": "invoice_extraction",
                        "schema": json.dumps(
                            response_schema
                        )
                    }
                }
            }
        },
        inferenceConfig={
            "temperature":
                APP_CONFIG["api"]["temperature"],
            "maxTokens":
                APP_CONFIG["api"]["max_tokens"]
        }
    )

    response_text = (
        response["output"]["message"]["content"][0]["text"]
        .strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
    )

    entities = json.loads(response_text)

    return {
        "file": name,
        "total_pages": len(pages),
        "entities": entities,
    }