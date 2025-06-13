import streamlit as st
import requests
from pathlib import Path
import os
from dotenv import load_dotenv
import json
from typing import Dict, Any
import pandas as pd

# Load environment variables
load_dotenv()

# Constants
BACKEND_URL = "http://localhost:8000"
API_V1_STR = "/api/v1"

def display_error(error_msg: str):
    """Display error message in a consistent format"""
    st.error(f"❌ {error_msg}")

def display_success(msg: str):
    """Display success message in a consistent format"""
    st.success(f"✅ {msg}")

def upload_files(files):
    """Upload files to the backend"""
    if not files:
        display_error("Please select at least one file to upload")
        return None

    try:
        files_to_upload = [("files", file) for file in files]
        response = requests.post(f"{BACKEND_URL}/upload", files=files_to_upload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        display_error("Could not connect to the backend server. Please make sure it's running.")
        return None
    except requests.exceptions.Timeout:
        display_error("Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        try:
            error_detail = e.response.json().get('detail', str(e))
        except:
            error_detail = str(e)
        display_error(f"Error uploading files: {error_detail}")
        return None

def display_upload_results(result):
    """Display upload results in a structured way"""
    if not result:
        return

    # Display overall status
    total_files = result.get('total_files', 0)
    processed_files = result.get('files_processed', 0)
    
    if processed_files == total_files:
        display_success(f"Successfully processed all {processed_files} files!")
    else:
        st.warning(f"Processed {processed_files} out of {total_files} files")

    # Display individual file results
    for file_result in result.get('results', []):
        with st.expander(f"File: {file_result['filename']}", expanded=True):
            if file_result['status'] == 'success':
                st.info(f"Status: Success")
                st.text(f"Number of chunks: {file_result['num_chunks']}")
                st.text("Text Preview:")
                st.text(file_result['extracted_text_preview'])
            else:
                st.error(f"Status: Failed")
                st.text(f"Error: {file_result.get('error', 'Unknown error')}")

def search_documents(query: str, top_k: int = 5, min_relevance: float = 0.3) -> Dict[str, Any]:
    """Search documents using the backend API"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "query": query,
                "top_k": top_k,
                "min_relevance_score": min_relevance,
                "include_context": True
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        try:
            error_detail = e.response.json().get('detail', str(e))
        except:
            error_detail = str(e)
        display_error(f"Error searching documents: {error_detail}")
        return None

def display_search_results(results: Dict[str, Any]):
    """Display search results in a structured way"""
    if not results:
        return

    st.subheader("Search Results")
    st.text(f"Query: {results['query']}")
    st.text(f"Processed query: {results['processed_query']}")
    st.text(f"Total matches: {results['total_matches']}")

    # Display theme summary if present
    if results.get('theme_summary'):
        st.markdown("### Synthesized Themes")
        st.info(results['theme_summary'])

    # Display results table if present
    if results.get('results_table'):
        df = pd.DataFrame(results['results_table'])
        st.markdown("### Results Table")
        st.dataframe(df)

    for i, result in enumerate(results['results'], 1):
        with st.expander(f"Result {i} (Score: {result['relevance_score']:.2f})", expanded=True):
            st.markdown("**Matched Text:**")
            st.text(result['chunk'])
            
            if result['context']:
                st.markdown("**Context:**")
                st.text(result['context'])
            
            st.markdown("**Details:**")
            st.text(f"Position: {result['position']}")
            st.text(f"Distance: {result['distance']:.4f}")

def main():
    st.title("Document Research & Theme Identification Chatbot")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "doc", "docx", "jpg", "jpeg", "png"]
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    result = upload_files(uploaded_files)
                    display_upload_results(result)
    
    # Main content area
    st.header("Ask Questions")
    
    # Search settings
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", 1, 20, 5)
    with col2:
        min_relevance = st.slider("Minimum relevance score", 0.0, 1.0, 0.3, 0.1)
    
    query = st.text_input("Enter your question:")
    
    if query:
        if st.button("Search"):
            with st.spinner("Searching documents..."):
                results = search_documents(query, top_k, min_relevance)
                display_search_results(results)

if __name__ == "__main__":
    main() 