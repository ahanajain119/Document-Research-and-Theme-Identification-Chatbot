from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import upload
from app.api import query
from app.api import query_results

app = FastAPI(
    title="Document Research & Theme Identification Chatbot",
    description="API for document processing, theme identification, and question answering",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(query.router)
app.include_router(query_results.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Document Research & Theme Identification Chatbot API",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }