import os
import json
from openai import OpenAI
from typing import Dict, List, Any
import logging

class QueryEngine:
    """Handles query processing and response generation using OpenAI"""
    
    def __init__(self):
        """Initialize the query engine with OpenAI client"""
        api_key = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
        self.client = OpenAI(api_key=api_key)
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        self.model = "gpt-4o"
    
    def process_query(self, query: str, query_type: str, relevant_chunks: List[Dict], documents: Dict) -> Dict:
        """
        Process a user query and generate a response
        
        Args:
            query: The user's question
            query_type: Type of query (e.g., "Direct Content Lookup", "Cross-Paper Comparison")
            relevant_chunks: List of relevant document chunks from RAG
            documents: Dictionary of all processed documents
            
        Returns:
            Dictionary containing the answer and sources
        """
        try:
            # Prepare context from relevant chunks
            context = self._prepare_context(relevant_chunks)
            
            # Generate response based on query type
            if query_type == "Cross-Paper Comparison":
                response = self._handle_comparison_query(query, context, relevant_chunks)
            elif query_type == "Methodology Summary":
                response = self._handle_methodology_query(query, context, documents)
            elif query_type == "Results Extraction":
                response = self._handle_results_query(query, context, relevant_chunks)
            elif query_type == "Conclusion Summary":
                response = self._handle_conclusion_query(query, context, documents)
            else:
                response = self._handle_general_query(query, context, relevant_chunks)
            
            return {
                'answer': response,
                'sources': self._format_sources(relevant_chunks),
                'query_type': query_type,
                'num_sources': len(relevant_chunks)
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your query: {str(e)}",
                'sources': [],
                'query_type': query_type,
                'num_sources': 0
            }
    
    def _prepare_context(self, relevant_chunks: List[Dict]) -> str:
        """Prepare context string from relevant chunks"""
        if not relevant_chunks:
            return "No relevant context found."
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks[:5], 1):  # Limit to top 5 chunks
            document = chunk.get('document', 'Unknown')
            page = chunk.get('page', 'N/A')
            text = chunk.get('text', '')
            score = chunk.get('similarity_score', 0)
            
            context_parts.append(
                f"[Source {i}] From {document} (Page {page}, Relevance: {score:.3f}):\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _handle_general_query(self, query: str, context: str, relevant_chunks: List[Dict]) -> str:
        """Handle general questions"""
        system_prompt = """You are an AI research assistant specialized in analyzing academic papers. 
        Your task is to answer questions based on the provided context from research papers.
        
        Guidelines:
        - Provide accurate, well-structured answers based on the context
        - Cite specific papers when referencing information
        - If information is not available in the context, clearly state this
        - Be concise but comprehensive
        - Use academic language appropriate for research contexts"""
        
        user_prompt = f"""Based on the following context from research papers, please answer this question:

Question: {query}

Context:
{context}

Please provide a detailed answer based on the available information."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _handle_comparison_query(self, query: str, context: str, relevant_chunks: List[Dict]) -> str:
        """Handle cross-paper comparison queries"""
        documents = list(set(chunk.get('document', '') for chunk in relevant_chunks))
        
        system_prompt = """You are an AI research assistant specialized in comparing research papers.
        Your task is to provide detailed comparisons between different papers based on the provided context.
        
        Guidelines:
        - Compare methodologies, results, approaches, and conclusions across papers
        - Highlight similarities and differences
        - Provide specific examples from each paper
        - Structure your comparison clearly
        - Note any limitations in the available information"""
        
        user_prompt = f"""Please compare the following research papers based on this question:

Question: {query}

Available papers: {', '.join(documents)}

Context from papers:
{context}

Provide a structured comparison addressing the question."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating comparison: {str(e)}"
    
    def _handle_methodology_query(self, query: str, context: str, documents: Dict) -> str:
        """Handle methodology-specific queries"""
        # Extract methodology sections from documents
        methodology_sections = []
        for doc_name, doc_data in documents.items():
            methodology = doc_data.get('structured_content', {}).get('methodology', '')
            if methodology:
                methodology_sections.append(f"From {doc_name}:\n{methodology}")
        
        system_prompt = """You are an AI research assistant specialized in analyzing research methodologies.
        Your task is to summarize and explain methodological approaches from research papers.
        
        Guidelines:
        - Focus on experimental design, data collection, and analysis methods
        - Explain technical approaches in clear language
        - Highlight innovative or unique methodological contributions
        - Compare different approaches when multiple papers are involved
        - Note any methodological limitations or assumptions"""
        
        full_context = context
        if methodology_sections:
            full_context += "\n\nSpecific Methodology Sections:\n" + "\n\n".join(methodology_sections)
        
        user_prompt = f"""Please analyze and summarize the methodology based on this question:

Question: {query}

Context:
{full_context}

Provide a comprehensive summary of the methodological approaches."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing methodology: {str(e)}"
    
    def _handle_results_query(self, query: str, context: str, relevant_chunks: List[Dict]) -> str:
        """Handle results and evaluation extraction queries"""
        system_prompt = """You are an AI research assistant specialized in extracting and analyzing research results.
        Your task is to identify and summarize key findings, evaluation metrics, and experimental results.
        
        Guidelines:
        - Extract specific numerical results, metrics, and performance measures
        - Identify evaluation criteria and benchmarks used
        - Summarize key findings and their significance
        - Compare results across different experiments or papers when applicable
        - Note any statistical significance or confidence intervals mentioned"""
        
        user_prompt = f"""Please extract and analyze the results based on this question:

Question: {query}

Context:
{context}

Focus on specific results, metrics, and evaluation outcomes."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error extracting results: {str(e)}"
    
    def _handle_conclusion_query(self, query: str, context: str, documents: Dict) -> str:
        """Handle conclusion and summary queries"""
        # Extract conclusion sections from documents
        conclusion_sections = []
        for doc_name, doc_data in documents.items():
            conclusion = doc_data.get('structured_content', {}).get('conclusion', '')
            if conclusion:
                conclusion_sections.append(f"From {doc_name}:\n{conclusion}")
        
        system_prompt = """You are an AI research assistant specialized in analyzing research conclusions.
        Your task is to summarize key conclusions, implications, and future work from research papers.
        
        Guidelines:
        - Summarize main conclusions and their significance
        - Identify key contributions and implications
        - Note future work suggestions and limitations
        - Highlight practical applications or theoretical advances
        - Synthesize conclusions across multiple papers when applicable"""
        
        full_context = context
        if conclusion_sections:
            full_context += "\n\nSpecific Conclusion Sections:\n" + "\n\n".join(conclusion_sections)
        
        user_prompt = f"""Please analyze and summarize the conclusions based on this question:

Question: {query}

Context:
{full_context}

Provide a comprehensive summary of the key conclusions and implications."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1200,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing conclusions: {str(e)}"
    
    def _format_sources(self, relevant_chunks: List[Dict]) -> List[Dict]:
        """Format source information for display"""
        sources = []
        for chunk in relevant_chunks:
            sources.append({
                'document': chunk.get('document', 'Unknown'),
                'page': chunk.get('page', 'N/A'),
                'content': chunk.get('text', '')[:300] + "..." if len(chunk.get('text', '')) > 300 else chunk.get('text', ''),
                'score': chunk.get('similarity_score', 0)
            })
        return sources
