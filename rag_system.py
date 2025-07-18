import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import logging

class RAGSystem:
    """Retrieval-Augmented Generation system for document search and retrieval"""
    
    def __init__(self, model_name: str = 'tfidf'):
        """
        Initialize the RAG system
        
        Args:
            model_name: Type of vectorization to use ('tfidf' for now)
        """
        # Enhanced TF-IDF for better accuracy
        self.vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased vocabulary size
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better context
            max_df=0.9,  # Less aggressive filtering
            min_df=1,    # Include rare terms that might be important
            sublinear_tf=True,  # Use log scaling
            norm='l2'    # L2 normalization
        )
        
        # Backup keyword-based search
        self.keyword_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 1),  # Only unigrams for exact matching
            stop_words=None,     # Keep all words for exact matching
            min_df=1,
            token_pattern=r'\b\w+\b'
        )
        
        self.document_chunks = []
        self.chunk_embeddings = None
        self.keyword_embeddings = None
        self.embedding_dimension = None
        self.fitted = False
        self.keyword_fitted = False
    
    def add_document(self, document_data: Dict) -> None:
        """
        Add a processed document to the RAG system
        
        Args:
            document_data: Dictionary containing document chunks and metadata
        """
        try:
            chunks = document_data.get('chunks', [])
            if not chunks:
                logging.warning(f"No chunks found in document {document_data.get('filename', 'unknown')}")
                return
            
            # Extract text from chunks
            chunk_texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings for the chunks using enhanced TF-IDF
            if not self.fitted:
                # Fit both vectorizers on all current text
                all_texts = [chunk['text'] for chunk in self.document_chunks] + chunk_texts
                self.vectorizer.fit(all_texts)
                self.keyword_vectorizer.fit(all_texts)
                self.fitted = True
                self.keyword_fitted = True
                
                # Re-vectorize existing chunks if any
                if self.document_chunks:
                    existing_texts = [chunk['text'] for chunk in self.document_chunks]
                    existing_embeddings = self.vectorizer.transform(existing_texts).toarray()
                    existing_keyword_embeddings = self.keyword_vectorizer.transform(existing_texts).toarray()
                    for i, chunk in enumerate(self.document_chunks):
                        chunk['embedding'] = existing_embeddings[i]
                        chunk['keyword_embedding'] = existing_keyword_embeddings[i]
            
            chunk_embeddings = self.vectorizer.transform(chunk_texts).toarray()
            keyword_embeddings = self.keyword_vectorizer.transform(chunk_texts).toarray()
            
            # Store chunk data with additional metadata
            for i, chunk in enumerate(chunks):
                enhanced_chunk = {
                    **chunk,
                    'embedding': chunk_embeddings[i],
                    'keyword_embedding': keyword_embeddings[i],
                    'document_title': document_data.get('structured_content', {}).get('title', ''),
                    'page': self._estimate_page_number(chunk, document_data),
                    'preprocessed_text': self._preprocess_text_for_search(chunk['text'])
                }
                self.document_chunks.append(enhanced_chunk)
            
            # Update the combined embeddings matrix
            self._update_embeddings_matrix()
            
            logging.info(f"Added {len(chunks)} chunks from {document_data.get('filename', 'unknown')}")
            
        except Exception as e:
            logging.error(f"Error adding document to RAG system: {str(e)}")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Enhanced retrieval with multiple strategies for better accuracy
        
        Args:
            query: The search query
            top_k: Number of top chunks to return
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if not self.document_chunks or self.chunk_embeddings is None:
            return []
        
        try:
            if not self.fitted:
                return []
            
            # Strategy 1: Semantic TF-IDF search
            semantic_results = self._semantic_search(query, top_k * 2)
            
            # Strategy 2: Keyword-based exact matching
            keyword_results = self._keyword_search(query, top_k)
            
            # Strategy 3: Fuzzy text matching
            fuzzy_results = self._fuzzy_text_search(query, top_k)
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, fuzzy_results, top_k
            )
            
            return combined_results
            
        except Exception as e:
            logging.error(f"Error retrieving relevant chunks: {str(e)}")
            return []
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Semantic search using enhanced TF-IDF"""
        query_embedding = self.vectorizer.transform([query]).toarray()[0]
        similarities = cosine_similarity([query_embedding], self.chunk_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            chunk = self.document_chunks[idx].copy()
            chunk['semantic_score'] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict]:
        """Exact keyword matching search"""
        if not self.keyword_fitted:
            return []
            
        query_embedding = self.keyword_vectorizer.transform([query]).toarray()[0]
        keyword_embeddings = np.array([chunk['keyword_embedding'] for chunk in self.document_chunks])
        similarities = cosine_similarity([query_embedding], keyword_embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        
        for idx in top_indices:
            chunk = self.document_chunks[idx].copy()
            chunk['keyword_score'] = float(similarities[idx])
            results.append(chunk)
        
        return results
    
    def _fuzzy_text_search(self, query: str, top_k: int) -> List[Dict]:
        """Fuzzy text matching for exact phrases"""
        import re
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        results = []
        
        for chunk in self.document_chunks:
            text_words = set(re.findall(r'\b\w+\b', chunk['text'].lower()))
            
            # Calculate word overlap
            overlap = len(query_words.intersection(text_words))
            total_words = len(query_words.union(text_words))
            
            if total_words > 0:
                jaccard_score = overlap / total_words
                
                # Boost score if exact phrases are found
                phrase_boost = 0
                if query.lower() in chunk['text'].lower():
                    phrase_boost = 0.5
                
                chunk_copy = chunk.copy()
                chunk_copy['fuzzy_score'] = jaccard_score + phrase_boost
                results.append(chunk_copy)
        
        # Sort by fuzzy score and return top results
        results.sort(key=lambda x: x['fuzzy_score'], reverse=True)
        return results[:top_k]
    
    def _combine_search_results(self, semantic: List[Dict], keyword: List[Dict], 
                               fuzzy: List[Dict], top_k: int) -> List[Dict]:
        """Combine and rank results from multiple search strategies"""
        chunk_scores = {}
        
        # Weight the different search strategies
        semantic_weight = 0.4
        keyword_weight = 0.3
        fuzzy_weight = 0.3
        
        # Aggregate scores for each chunk
        for chunk in semantic:
            chunk_id = chunk['id']
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'chunk': chunk, 'total_score': 0, 'methods': []}
            
            score = chunk.get('semantic_score', 0) * semantic_weight
            chunk_scores[chunk_id]['total_score'] += score
            chunk_scores[chunk_id]['methods'].append(f"semantic: {chunk.get('semantic_score', 0):.3f}")
        
        for chunk in keyword:
            chunk_id = chunk['id']
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'chunk': chunk, 'total_score': 0, 'methods': []}
            
            score = chunk.get('keyword_score', 0) * keyword_weight
            chunk_scores[chunk_id]['total_score'] += score
            chunk_scores[chunk_id]['methods'].append(f"keyword: {chunk.get('keyword_score', 0):.3f}")
        
        for chunk in fuzzy:
            chunk_id = chunk['id']
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {'chunk': chunk, 'total_score': 0, 'methods': []}
            
            score = chunk.get('fuzzy_score', 0) * fuzzy_weight
            chunk_scores[chunk_id]['total_score'] += score
            chunk_scores[chunk_id]['methods'].append(f"fuzzy: {chunk.get('fuzzy_score', 0):.3f}")
        
        # Sort by total score and return top results
        sorted_results = sorted(chunk_scores.values(), key=lambda x: x['total_score'], reverse=True)
        
        final_results = []
        for item in sorted_results[:top_k]:
            chunk = item['chunk'].copy()
            chunk['similarity_score'] = item['total_score']
            chunk['search_methods'] = item['methods']
            
            # Clean up embeddings to save memory
            for key in ['embedding', 'keyword_embedding']:
                if key in chunk:
                    del chunk[key]
            
            final_results.append(chunk)
        
        return final_results
    
    def _preprocess_text_for_search(self, text: str) -> str:
        """Preprocess text for better search accuracy"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Preserve important punctuation but clean others
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        
        return text.strip()
    
    def semantic_search(self, query: str, document_filter: str = None, threshold: float = 0.3) -> List[Dict]:
        """
        Perform semantic search with optional document filtering
        
        Args:
            query: Search query
            document_filter: Optional document name to filter results
            threshold: Minimum similarity threshold
            
        Returns:
            List of relevant chunks above threshold
        """
        if not self.document_chunks:
            return []
        
        try:
            # Get all relevant chunks
            all_chunks = self.retrieve_relevant_chunks(query, top_k=len(self.document_chunks))
            
            # Filter by document if specified
            if document_filter:
                all_chunks = [chunk for chunk in all_chunks if chunk['document'] == document_filter]
            
            # Filter by threshold
            relevant_chunks = [chunk for chunk in all_chunks if chunk['similarity_score'] >= threshold]
            
            return relevant_chunks
            
        except Exception as e:
            logging.error(f"Error in semantic search: {str(e)}")
            return []
    
    def get_document_summary(self, document_name: str) -> Dict:
        """
        Get summary information for a specific document
        
        Args:
            document_name: Name of the document
            
        Returns:
            Dictionary with document summary
        """
        document_chunks = [chunk for chunk in self.document_chunks if chunk['document'] == document_name]
        
        if not document_chunks:
            return {}
        
        return {
            'document': document_name,
            'total_chunks': len(document_chunks),
            'total_characters': sum(len(chunk['text']) for chunk in document_chunks),
            'title': document_chunks[0].get('document_title', ''),
            'chunk_ids': [chunk['id'] for chunk in document_chunks]
        }
    
    def compare_documents(self, doc1: str, doc2: str, query: str) -> Dict:
        """
        Compare two documents based on a specific query
        
        Args:
            doc1: First document name
            doc2: Second document name
            query: Comparison query
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Get relevant chunks from each document
            doc1_chunks = self.semantic_search(query, document_filter=doc1, threshold=0.2)
            doc2_chunks = self.semantic_search(query, document_filter=doc2, threshold=0.2)
            
            return {
                'query': query,
                'document1': {
                    'name': doc1,
                    'relevant_chunks': doc1_chunks[:3],  # Top 3 chunks
                    'max_similarity': max([chunk['similarity_score'] for chunk in doc1_chunks]) if doc1_chunks else 0
                },
                'document2': {
                    'name': doc2,
                    'relevant_chunks': doc2_chunks[:3],  # Top 3 chunks
                    'max_similarity': max([chunk['similarity_score'] for chunk in doc2_chunks]) if doc2_chunks else 0
                }
            }
            
        except Exception as e:
            logging.error(f"Error comparing documents: {str(e)}")
            return {}
    
    def _update_embeddings_matrix(self) -> None:
        """Update the combined embeddings matrix"""
        if not self.document_chunks:
            self.chunk_embeddings = None
            return
        
        embeddings = [chunk['embedding'] for chunk in self.document_chunks]
        self.chunk_embeddings = np.array(embeddings)
        self.embedding_dimension = self.chunk_embeddings.shape[1] if len(embeddings) > 0 else None
    
    def _estimate_page_number(self, chunk: Dict, document_data: Dict) -> int:
        """
        Estimate page number for a chunk based on position
        
        Args:
            chunk: Chunk dictionary
            document_data: Full document data
            
        Returns:
            Estimated page number
        """
        try:
            total_pages = document_data.get('total_pages', 1)
            full_text_length = len(document_data.get('full_text', ''))
            
            if full_text_length == 0:
                return 1
            
            chunk_start = chunk.get('start_pos', 0)
            page_estimate = max(1, int((chunk_start / full_text_length) * total_pages))
            
            return min(page_estimate, total_pages)
            
        except:
            return 1
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG system"""
        documents = set(chunk['document'] for chunk in self.document_chunks)
        
        return {
            'total_chunks': len(self.document_chunks),
            'total_documents': len(documents),
            'embedding_dimension': self.embedding_dimension,
            'documents': list(documents)
        }
