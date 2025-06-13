import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initialize the vector store with ChromaDB and Google's Generative AI embeddings.
        
        Args:
            persist_directory (str): Directory to persist the ChromaDB database
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Initialize embeddings model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )
        
        # Set up persist directory
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load the vector store
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> Chroma:
        """
        Initialize or load the ChromaDB vector store.
        """
        try:
            # Try to load existing vector store
            return Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
        except Exception:
            # If loading fails, create a new one
            return Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents (List[Document]): List of documents to add
        """
        self.vector_store.add_documents(documents)
        self.vector_store.persist()
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            filter (Optional[dict]): Filter criteria for the search
            
        Returns:
            List[Document]: List of similar documents
        """
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.
        
        Returns:
            int: Number of documents
        """
        return self.vector_store._collection.count()
    
    def delete_documents(self, filter: dict) -> None:
        """
        Delete documents from the vector store based on filter criteria.
        
        Args:
            filter (dict): Filter criteria for documents to delete
        """
        self.vector_store._collection.delete(where=filter)
        self.vector_store.persist()
    
    def clear(self) -> None:
        """
        Clear all documents from the vector store.
        """
        self.vector_store._collection.delete(where={})
        self.vector_store.persist()
    
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents from the vector store.
        
        Returns:
            List[Document]: List of all documents
        """
        return self.vector_store.get() 