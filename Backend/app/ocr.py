import pytesseract
from PIL import Image
import docx2txt
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to set Tesseract path, but don't fail if not found
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except Exception as e:
    logger.warning(f"Could not set Tesseract path: {e}")

def extract_text_from_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext in [".png", ".jpg", ".jpeg"]:
            try:
                image = Image.open(filepath)
                text = pytesseract.image_to_string(image)
                if not text.strip():
                    logger.warning(f"No text extracted from image {filepath}")
                    return ""
            except Exception as e:
                logger.error(f"Error processing image {filepath}: {e}")
                return ""
        elif ext == ".docx":
            try:
                text = docx2txt.process(filepath)
                if not text.strip():
                    logger.warning(f"No text extracted from docx {filepath}")
                    return ""
            except Exception as e:
                logger.error(f"Error processing docx {filepath}: {e}")
                return ""
        else:
            logger.warning(f"Unsupported file type for OCR: {ext}")
            return ""
    except Exception as e:
        logger.error(f"Unexpected error in extract_text_from_file for {filepath}: {e}")
        return ""

    return text
