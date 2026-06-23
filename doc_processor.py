# from pathlib import Path
import os
import io
import pymupdf  # PyMuPDF
from PIL import Image
from typing import Optional, List, Dict, Any



OUTPUT_DIR = "./saved_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_pdf(file_path: str, password: Optional[str] = None) -> List[Dict[str, Any]]:
    """Opens a PDF, authenticates with password if needed, saves pages as PNGs,

    and returns raw image bytes for downstream OCR/LLM tasks.
    """
    processed_pages = []
    filename = os.path.basename(file_path)
    
    # Open the document
    doc = pymupdf.open(file_path)
    
    # Handle password protection
    if doc.is_encrypted:
        if not password:
            raise ValueError(f"The PDF '{filename}' is encrypted. A password is required.")
        if not doc.authenticate(password):
            raise ValueError(f"Incorrect password provided for the PDF '{filename}'.")
            
    # Iterate through all pages
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        
        # Scale resolution by 3x (raises default 72 DPI to ~216 DPI)
        # This prevents text pixelation and ensures high downstream OCR accuracy
        matrix = pymupdf.Matrix(3, 3)
        pix = page.get_pixmap(matrix=matrix)
        
        # Save PNG file to disk
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}_page_{page_num + 1}.png"
        os.makedirs(os.path.join(OUTPUT_DIR, base_name), exist_ok = True)
        save_path = os.path.join(OUTPUT_DIR, base_name, output_filename)
        pix.save(save_path)
        
        # Extract raw PNG bytes to avoid reloading the file from disk later
        png_bytes = pix.tobytes("png")
        
        processed_pages.append({
            "page_number": page_num + 1,
            "saved_path": save_path,
            "png_bytes": png_bytes  # Pass this byte array directly to your LLM/OCR
        })
        
    doc.close()
    return processed_pages


def process_tiff(file_path: str) -> List[Dict[str, Any]]:
    """Opens a single or multi-page TIFF file using Pillow, saves pages as PNGs,

    and returns raw image bytes.
    """
    processed_pages = []
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    
    # Open TIFF file using Pillow
    img = Image.open(file_path)
    page_num = 1

    while True:
        output_filename = f"{base_name}_page_{page_num}.png"
        save_path = os.path.join(OUTPUT_DIR, base_name, output_filename)
        os.makedirs(os.path.join(OUTPUT_DIR, base_name), exist_ok = True)
        
        # Convert non-RGB profiles (like CMYK or Palette profiles) to standard RGB
        if img.mode not in ("RGB", "RGBA"):
            converted_img = img.convert("RGB")
        else:
            converted_img = img
            
        # Save PNG file to disk
        converted_img.save(save_path, format="PNG")
        
        # Extract raw PNG bytes in memory
        buffer = io.BytesIO()
        converted_img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        
        processed_pages.append({
            "page_number": page_num,
            "saved_path": save_path,
            "png_bytes": png_bytes  # Pass this byte array directly to your LLM/OCR
        })
        
        # Move to the next page frame if a multi-page TIFF exists
        try:
            img.seek(img.tell() + 1)
            page_num += 1
        except EOFError:
            break  # End of the TIFF frames
            
    return processed_pages


def convert_document(file_path: str, password: Optional[str] = None) -> List[Dict[str, Any]]:
    """Helper Router function to handle input based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return process_pdf(file_path, password)
    elif ext in (".tiff", ".tif"):
        return process_tiff(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF and TIFF are supported.")




if __name__ == "__main__":
    try:

        # Specify the directory path
        directory_path = Path(r"test")
        # Loop through all items in the directory
        for item in directory_path.iterdir():
            if item.is_file():
                print(f"File: {item.name} | Full Path: {item}")


            # pdf_results = convert_document("secure_invoice.pdf", password="mypassword123")
            tiff_results = convert_document(item) #1008_1234.pdf, Invoices\DPSF\00724377.TIF

            print(f"Successfully processed {len(tiff_results)} pages.")
            print(f"First page saved at: {tiff_results[0]['saved_path']}")
            print(f"In-memory byte size: {len(tiff_results[0]['png_bytes'])} bytes")

    except Exception as e:
        print(f"Error processing file: {e}")
