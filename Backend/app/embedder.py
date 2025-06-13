import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the embedding model 
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text_chunks(chunks):
    """Embed text chunks and create FAISS index"""
    try:
        # Extract text from chunk dictionaries
        texts = [chunk["chunk_text"] for chunk in chunks]
        logger.info(f"Embedding {len(texts)} text chunks")
        
        # Generate embeddings
        embeddings = model.encode(texts, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Create and populate FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        logger.info("FAISS index created and populated")
        
        return index, embeddings
    except Exception as e:
        logger.error(f"Error in embed_text_chunks: {str(e)}")
        raise

# Save and load the FAISS index and chunks
INDEX_PATH = "faiss_index.index"
CHUNKS_PATH = "chunks.pkl"

def save_index(index, chunks):
    """Save FAISS index and chunks to disk"""
    try:
        logger.info(f"Saving FAISS index to {INDEX_PATH}")
        faiss.write_index(index, INDEX_PATH)
        
        logger.info(f"Saving chunks to {CHUNKS_PATH}")
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)
        
        logger.info("Successfully saved index and chunks")
    except Exception as e:
        logger.error(f"Error saving index and chunks: {str(e)}")
        raise

def load_index_and_chunks():
    """Load FAISS index and chunks from disk"""
    try:
        if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            logger.warning("Index or chunks file not found")
            return None, None
            
        logger.info(f"Loading FAISS index from {INDEX_PATH}")
        index = faiss.read_index(INDEX_PATH)
        
        logger.info(f"Loading chunks from {CHUNKS_PATH}")
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
            
        logger.info("Successfully loaded index and chunks")
        return index, chunks
    except Exception as e:
        logger.error(f"Error loading index and chunks: {str(e)}")
        return None, None

def create_faiss_index(chunks):
    """Create a new FAISS index from chunks"""
    try:
        logger.info("Creating new FAISS index")
        index, embeddings = embed_text_chunks(chunks)
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise
