from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
from app.embedder import load_index_and_chunks
from sentence_transformers import SentenceTransformer

router = APIRouter()

model = SentenceTransformer('all-MiniLM-L6-v2')

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5  # number of chunks to retrieve

@router.post("/query_results")
async def query_documents_formatted(request: QueryRequest):
    index, chunks = load_index_and_chunks()
    if index is None or chunks is None:
        raise HTTPException(status_code=400, detail="No indexed documents found. Upload files first.")

    query_embedding = model.encode([request.query])[0]
    D, I = index.search(np.array([query_embedding]), request.top_k)

    results_table = []
    for rank, idx in enumerate(I[0]):
        if idx >= len(chunks):
            continue
    #     Retrieve the chunk and its citation information
        chunk = chunks[idx]
        citation = {
        "document": chunk["document"],
        "page": chunk["page"],
        "paragraph": chunk["paragraph"],
        "sentence": chunk["sentence"]
        }
        # Format for the results table
        results_table.append({
        "Document": citation["document"],
        "Extracted Answer": chunk["chunk_text"][:300] + ("..." if len(chunk["chunk_text"]) > 300 else ""),
        "Citation": f"Page {citation['page']}, Para {citation['paragraph']}, Sent {citation['sentence']}"
        })


    # Placeholder synthesized theme summary (replace with your LLM or logic)
    synthesized_themes = {
        "Theme 1 – Regulatory Non-Compliance": "DOC001, DOC002: Highlight regulatory non-compliance with SEBI Act and LODR.",
        "Theme 2 – Penalty Justification": "DOC001: Explicit justification of penalties under statutory frameworks."
    }

    return {
        "query": request.query,
        "results_table": results_table,
        "synthesized_theme_summary": synthesized_themes
    }
