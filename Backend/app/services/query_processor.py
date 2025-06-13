from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class QueryProcessor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
        # Load environment variables
        load_dotenv()
        
        # Get OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        # Initialize OpenAI client with API key
        self.openai_client = OpenAI(api_key=api_key)
        
        # Common English stopwords
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
            'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
            'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'than', 'too', 'very', 'can', 'will',
            'just', 'should', 'now'
        }
        
        # Medical terminology mappings
        self.medical_terms = {
            'pee': ['urine', 'urination', 'urinary', 'micturition'],
            'pink': ['reddish', 'blood-tinged', 'hematuria', 'blood in urine'],
            'urine': ['pee', 'micturition', 'urination'],
            'blood': ['hematuria', 'blood-tinged', 'reddish'],
        }

    def preprocess_query(self, query: str) -> str:
        """Clean and preprocess the query text"""
        # Convert to lowercase
        query = query.lower()
        
        # Remove special characters
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Split into words using regex
        tokens = re.findall(r'\b\w+\b', query)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Join tokens back into string
        return ' '.join(tokens)

    def expand_query(self, query: str) -> List[str]:
        """Use LLM to intelligently expand the query with relevant variations"""
        try:
            # Create a prompt for the LLM
            prompt = f"""Given the user query: "{query}"
            Generate a list of alternative phrasings and medical/technical terms that would help find relevant information.
            Consider:
            1. Medical/technical terminology
            2. Common variations of the query
            3. Related symptoms or conditions
            4. Different ways to express the same concept
            
            Return only a comma-separated list of variations, nothing else."""

            # Get variations from LLM
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a medical information retrieval assistant. Your task is to expand user queries with relevant medical terminology and variations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            # Process the response
            variations_text = response.choices[0].message.content.strip()
            variations = [v.strip() for v in variations_text.split(',')]
            
            # Add original query and its preprocessed version
            variations = [query, self.preprocess_query(query)] + variations
            
            # Remove duplicates and empty strings
            return list(set(filter(None, variations)))
            
        except Exception as e:
            logger.error(f"Error expanding query with LLM: {str(e)}")
            # Fallback to basic medical terminology expansion
            variations = [query, self.preprocess_query(query)]
            
            # Basic medical term mappings
            medical_terms = {
                'pee': ['urine', 'urination', 'urinary'],
                'pink': ['reddish', 'blood-tinged', 'hematuria'],
                'blood': ['hematuria', 'blood-tinged'],
                'urine': ['pee', 'micturition'],
                'pain': ['discomfort', 'ache', 'soreness'],
                'hurt': ['pain', 'discomfort', 'ache'],
                'sick': ['ill', 'unwell', 'diseased'],
                'throw up': ['vomit', 'nausea', 'emesis'],
                'fever': ['pyrexia', 'temperature'],
                'rash': ['eruption', 'dermatitis', 'skin condition']
            }
            
            # Add variations based on medical terms
            words = query.lower().split()
            for word in words:
                if word in medical_terms:
                    variations.extend(medical_terms[word])
            
            return list(set(filter(None, variations)))

    def calculate_relevance_score(self, query_embedding: np.ndarray, 
                                chunk_embedding: np.ndarray, 
                                distance: float) -> float:
        """Calculate a normalized relevance score"""
        # Convert distance to similarity score (FAISS uses L2 distance)
        max_distance = 2.0  # Approximate maximum L2 distance for normalized embeddings
        similarity = 1 - (distance / max_distance)
        
        # Calculate cosine similarity
        cosine_sim = util.cos_sim(query_embedding, chunk_embedding)[0][0].item()
        
        # Combine scores (weighted average)
        combined_score = 0.7 * cosine_sim + 0.3 * similarity
        
        # Lower the threshold for medical queries
        return max(0.0, min(1.0, combined_score * 1.2))  # Boost the score by 20%

    def get_context_window(self, chunks: List[str], 
                         match_index: int, 
                         window_size: int = 2) -> str:
        """Get surrounding context for a matched chunk"""
        start_idx = max(0, match_index - window_size)
        end_idx = min(len(chunks), match_index + window_size + 1)
        
        context = chunks[start_idx:end_idx]
        return " ".join(context)

    def process_search_results(self, 
                             query: str,
                             chunks: List[Dict],
                             distances: List[float],
                             indices: List[int],
                             top_k: int = 5) -> List[Dict[str, Any]]:
        """Process and enhance search results"""
        query_embedding = self.model.encode([query])[0]
        
        results = []
        for i, (chunk, distance, idx) in enumerate(zip(chunks, distances, indices)):
            # Get the text content from the chunk dictionary
            chunk_text = chunk["chunk_text"]
            chunk_embedding = self.model.encode([chunk_text])[0]
            
            # Calculate relevance score
            relevance_score = self.calculate_relevance_score(
                query_embedding, chunk_embedding, distance
            )
            
            # Get context window
            context = self.get_context_window([c["chunk_text"] for c in chunks], idx)
            
            results.append({
                "chunk": chunk_text,
                "context": context,
                "relevance_score": relevance_score,
                "distance": float(distance),
                "position": idx
            })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return results[:top_k]

    def synthesize_themes(self, query: str, chunks: list) -> str:
        # Prepare the prompt
        chunk_texts = [
            f"- {chunk['chunk_text']} (Document: {chunk['document']}, Page: {chunk['page']}, Paragraph: {chunk['paragraph']})"
            for chunk in chunks
        ]
        prompt = f"""
Given the following extracted text chunks in response to the query: \"{query}\", group them into main themes.
For each theme, provide:
- A short, chat-style summary
- Supporting document citations (document name, page, paragraph)

Chunks:
{chr(10).join(chunk_texts)}

Return your answer as a list of themes, each with a summary and citations.
"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that synthesizes research themes from document chunks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error synthesizing themes: {str(e)}")
            return "Theme synthesis failed."