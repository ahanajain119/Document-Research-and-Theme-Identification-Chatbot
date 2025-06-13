from typing import List, Dict, Optional
import os
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PDFLoader
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, upload_dir: str):
        self.upload_dir = Path(upload_dir)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
    def process_document(self, file_path: str) -> List[Document]:
        """Process a document and return chunks of text with metadata."""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        # Extract text based on file type
        if file_extension == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            return self._process_image(file_path)
        elif file_extension == '.txt':
            return self._process_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process PDF files with OCR for scanned pages."""
        documents = []
        pdf = PdfReader(str(file_path))
        
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            # If no text is extracted, the page might be scanned
            if not text.strip():
                # Convert PDF page to image and perform OCR
                images = convert_from_path(str(file_path), first_page=page_num+1, last_page=page_num+1)
                text = pytesseract.image_to_string(images[0])
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create Document objects with metadata
            for chunk_num, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": str(file_path),
                        "page": page_num + 1,
                        "chunk": chunk_num + 1,
                        "file_type": "pdf"
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _process_image(self, file_path: Path) -> List[Document]:
        """Process image files using OCR."""
        # Perform OCR on the image
        text = pytesseract.image_to_string(str(file_path))
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for chunk_num, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": str(file_path),
                    "page": 1,
                    "chunk": chunk_num + 1,
                    "file_type": "image"
                }
            )
            documents.append(doc)
        
        return documents
    
    def _process_text(self, file_path: Path) -> List[Document]:
        """Process text files."""
        # Load and split text
        loader = TextLoader(str(file_path))
        documents = loader.load()
        
        # Split into chunks
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add additional metadata
        for doc in split_docs:
            doc.metadata.update({
                "file_type": "text",
                "source": str(file_path)
            })
        
        return split_docs 