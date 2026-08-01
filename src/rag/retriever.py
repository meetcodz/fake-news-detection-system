"""Lightweight Local RAG implementation using TF-IDF and Cosine Similarity."""

import json
import logging
from typing import Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

logger = logging.getLogger(__name__)

class LocalRAG:
    """Zero-dependency vector retriever using TF-IDF."""
    
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.documents: list[dict[str, Any]] = []
        self.tfidf_matrix = None
        self.is_fitted = False
        
    def add_documents_from_json(self, json_path: str | Path) -> None:
        """Load documents from a JSON file and build the search index."""
        path = Path(json_path)
        if not path.exists():
            logger.warning(f"RAG knowledge base not found at {path}")
            return
            
        with open(path, "r", encoding="utf-8") as f:
            docs = json.load(f)
            
        self.add_documents(docs)
        
    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """Add documents and build the TF-IDF matrix."""
        if not documents:
            return
            
        self.documents = documents
        
        # Combine title and content for better matching
        corpus = [f"{doc.get('title', '')} {doc.get('content', '')}" for doc in self.documents]
        
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True
        logger.info(f"LocalRAG initialized with {len(self.documents)} documents.")
        
    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.05) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant documents for the given query."""
        if not self.is_fitted or not query.strip():
            return []
            
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get indices of the top_k similarities
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= threshold:
                doc = self.documents[idx].copy()
                doc["similarity_score"] = float(score)
                results.append(doc)
                
        return results
