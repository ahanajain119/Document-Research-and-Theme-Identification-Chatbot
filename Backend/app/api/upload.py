import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.ocr import extract_text_from_file
from app.chunker import chunk_text
from app.embedder import save_index, create_faiss_index
import PyPDF2
from typing import List
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg"}

# Get the absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploaded_files")
PROCESSED_TEXT_DIR = os.path.join(BACKEND_DIR, "processed_text")

class FileProcessingError(Exception):
    """Custom exception for file processing errors"""
    pass

def validate_file(file: UploadFile) -> None:
    """Validate file size and type"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {ext} not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Reset file pointer
    file.file.seek(0)

async def process_file(file: UploadFile, upload_folder: str) -> dict:
    """Process a single file and return results"""
    try:
        logger.info(f"Starting to process file: {file.filename}")
        validate_file(file)
        
        # Create upload directory if it doesn't exist
        os.makedirs(upload_folder, exist_ok=True)
        
        # Save file
        file_location = os.path.join(upload_folder, file.filename)
        try:
            contents = await file.read()
            with open(file_location, "wb") as buffer:
                buffer.write(contents)
            logger.info(f"File saved successfully at: {file_location}")
        finally:
            await file.close()
        
        # Extract text based on file type
        extracted_text = ""
        ext = os.path.splitext(file.filename)[1].lower()
        
        try:
            if ext == ".pdf":
                logger.info(f"Processing PDF file: {file.filename}")
                with open(file_location, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        extracted_text += page.extract_text() + "\n"
            elif ext == ".txt":
                logger.info(f"Processing text file: {file.filename}")
                with open(file_location, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
            else:  # For other file types, use OCR
                logger.info(f"Processing file with OCR: {file.filename}")
                extracted_text = extract_text_from_file(file_location)
                if not extracted_text.strip():
                    raise FileProcessingError("OCR failed to extract text from the image")
            
            logger.info(f"Extracted text length: {len(extracted_text)} characters")
        except Exception as e:
            logger.error(f"Error extracting text from {file.filename}: {str(e)}")
            raise FileProcessingError(f"Failed to extract text: {str(e)}")

        # Chunk the extracted text
        try:
            logger.info(f"Starting text chunking for: {file.filename}")
            chunks = chunk_text(extracted_text, doc_name=file.filename)
            if not chunks:
                raise FileProcessingError("No text chunks generated")
            logger.info(f"Generated {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error chunking text from {file.filename}: {str(e)}")
            raise FileProcessingError(f"Failed to chunk text: {str(e)}")

        # Create FAISS index and save it
        try:
            logger.info(f"Creating FAISS index for: {file.filename}")
            index = create_faiss_index(chunks)
            save_index(index, chunks)
            logger.info("FAISS index created and saved successfully")
        except Exception as e:
            logger.error(f"Error creating index for {file.filename}: {str(e)}")
            raise FileProcessingError(f"Failed to create search index: {str(e)}")

        # Save the extracted text
        try:
            os.makedirs(PROCESSED_TEXT_DIR, exist_ok=True)
            text_file_path = os.path.join(PROCESSED_TEXT_DIR, os.path.splitext(file.filename)[0] + ".txt")
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(extracted_text)
            logger.info(f"Processed text saved to: {text_file_path}")
        except Exception as e:
            logger.error(f"Error saving processed text for {file.filename}: {str(e)}")
            raise FileProcessingError(f"Failed to save processed text: {str(e)}")

        return {
            "filename": file.filename,
            "status": "success",
            "num_chunks": len(chunks),
            "extracted_text_preview": extracted_text[:500]
        }

    except FileProcessingError as e:
        logger.error(f"File processing error for {file.filename}: {str(e)}")
        return {
            "filename": file.filename,
            "status": "error",
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {str(e)}")
        return {
            "filename": file.filename,
            "status": "error",
            "error": "An unexpected error occurred during processing"
        }

@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    for file in files:
        result = await process_file(file, UPLOAD_DIR)
        results.append(result)
    
    # Check if any files were processed successfully
    successful_uploads = [r for r in results if r["status"] == "success"]
    if not successful_uploads:
        raise HTTPException(
            status_code=400,
            detail="No files were processed successfully. Check the results for details."
        )
    
    return {
        "files_processed": len(successful_uploads),
        "total_files": len(files),
        "results": results
    }

