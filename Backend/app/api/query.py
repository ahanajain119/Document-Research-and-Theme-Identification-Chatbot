from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from app.embedder import load_index_and_chunks
from app.services.query_processor import QueryProcessor
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
query_processor = QueryProcessor()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    include_context: bool = True
    min_relevance_score: float = 0.3

class SearchResult(BaseModel):
    chunk: str
    context: str
    relevance_score: float
    distance: float
    position: int

class QueryResponse(BaseModel):
    query: str
    processed_query: str
    results: List[SearchResult]
    total_matches: int

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        # Load index and chunks
        index, chunks = load_index_and_chunks()
        if index is None or chunks is None:
            raise HTTPException(
                status_code=400, 
                detail="No indexed documents found. Upload files first."
            )

        # Preprocess and expand query
        processed_query = query_processor.preprocess_query(request.query)
        query_variations = query_processor.expand_query(request.query)
        
        # Get initial results from FAISS
        all_results = []
        for query_var in query_variations:
            query_embedding = query_processor.model.encode([query_var])[0]
            D, I = index.search(np.array([query_embedding]), request.top_k)
            
            # Process results
            results = query_processor.process_search_results(
                query=query_var,
                chunks=chunks,
                distances=D[0].tolist(),
                indices=I[0].tolist(),
                top_k=request.top_k
            )
            all_results.extend(results)

        # Remove duplicates and filter by relevance
        seen_positions = set()
        filtered_results = []
        for result in sorted(all_results, key=lambda x: x["relevance_score"], reverse=True):
            if (result["position"] not in seen_positions and 
                result["relevance_score"] >= request.min_relevance_score):
                seen_positions.add(result["position"])
                filtered_results.append(result)
                if len(filtered_results) >= request.top_k:
                    break

        # Convert to response model
        search_results = [
            SearchResult(
                chunk=result["chunk"],
                context=result["context"] if request.include_context else "",
                relevance_score=result["relevance_score"],
                distance=result["distance"],
                position=result["position"]
            )
            for result in filtered_results
        ]

        return QueryResponse(
            query=request.query,
            processed_query=processed_query,
            results=search_results,
            total_matches=len(filtered_results)
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )
