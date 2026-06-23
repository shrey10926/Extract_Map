from pathlib import Path
import os, io
from typing import Optional, List, Dict, Any
import fitz
from PIL import Image


OUTPUT_DIR = "./saved_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_image(
    img: Image.Image,
    output_path: str = None,
    max_edge: int = 1568
) -> bytes:
    """
    Convert image to RGB, resize if needed, save as optimized PNG bytes.
    """
    # Keep colour, normalize mode
    img = img.convert("RGB")

    # Resize so longest edge <= max_edge
    width, height = img.size
    if max(width, height) > max_edge:
        if width > height:
            new_width = max_edge
            new_height = int(height * (max_edge / width))
        else:
            new_height = max_edge
            new_width = int(width * (max_edge / height))

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Save to bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    img_bytes = buffer.getvalue()

    # Optionally save to disk
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(img_bytes)

    return img_bytes


def process_pdf(file_path: str, password: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Render each PDF page, preprocess it, and return PNG bytes.
    """
    processed_pages = []
    filename = os.path.basename(file_path)

    doc = fitz.open(file_path)

    try:
        if doc.is_encrypted:
            if not password:
                raise ValueError(f"The PDF '{filename}' is encrypted. A password is required.")
            if not doc.authenticate(password):
                raise ValueError(f"Incorrect password provided for the PDF '{filename}'.")

        base_name = os.path.splitext(filename)[0]
        out_dir = os.path.join(OUTPUT_DIR, base_name)
        os.makedirs(out_dir, exist_ok=True)

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Render at a reasonable resolution.
            # 2x to 3x is usually enough for invoices; 3x is safer for small text.
            matrix = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=matrix, alpha=False)

            # Convert pixmap directly to PIL image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            output_filename = f"{base_name}_page_{page_num + 1}.png"
            save_path = os.path.join(out_dir, output_filename)

            png_bytes = preprocess_image(
                img=img,
                output_path=save_path,
                max_edge=1568
            )

            processed_pages.append({
                "page_number": page_num + 1,
                "saved_path": save_path,
                "png_bytes": png_bytes
            })

    finally:
        doc.close()

    return processed_pages


def process_tiff(file_path: str) -> List[Dict[str, Any]]:
    """
    Process single or multi-page TIFF and return preprocessed PNG bytes.
    """
    processed_pages = []
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    out_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(file_path)
    page_num = 1

    while True:
        try:
            frame = img.copy()

            output_filename = f"{base_name}_page_{page_num}.png"
            save_path = os.path.join(out_dir, output_filename)

            png_bytes = preprocess_image(
                img=frame,
                output_path=save_path,
                max_edge=1568
            )

            processed_pages.append({
                "page_number": page_num,
                "saved_path": save_path,
                "png_bytes": png_bytes
            })

            page_num += 1
            img.seek(img.tell() + 1)

        except EOFError:
            break

    return processed_pages


def process_image_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process JPG/JPEG/PNG/WebP etc. as a single page image.
    """
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    out_dir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(file_path)

    output_filename = f"{base_name}.png"
    save_path = os.path.join(out_dir, output_filename)

    png_bytes = preprocess_image(
        img=img,
        output_path=save_path,
        max_edge=1568
    )

    return [{
        "page_number": 1,
        "saved_path": save_path,
        "png_bytes": png_bytes
    }]


def convert_document(file_path: str, password: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Route document processing based on file extension.
    """
    file_path = str(file_path)  # handles Path objects too
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return process_pdf(file_path, password=password)
    elif ext in (".tiff", ".tif"):
        return process_tiff(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        return process_image_file(file_path)
    else:
        raise ValueError(
            f"Unsupported file format: {ext}. Supported: PDF, TIFF, JPG, JPEG, PNG, WebP, BMP."
        )


if __name__ == "__main__":
    try:
        directory_path = Path(r"test")

        for item in directory_path.iterdir():
            if item.is_file():
                print(f"File: {item.name} | Full Path: {item}")

                results = convert_document(str(item))  # IMPORTANT: pass string path
                print(f"Successfully processed {len(results)} pages.")
                print(f"First page saved at: {results[0]['saved_path']}")
                print(f"In-memory byte size: {len(results[0]['png_bytes'])} bytes")

    except Exception as e:
        print(f"Error processing file: {e}")