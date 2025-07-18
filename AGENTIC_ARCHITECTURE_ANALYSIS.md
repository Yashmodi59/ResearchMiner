# Building Agentic-Like Systems Without LangChain/LangGraph

## Executive Summary

This document explains how we built ResearchMiner - an intelligent research paper analysis system that exhibits agentic-like behavior without using LangChain or LangGraph. We achieved this through custom Python implementation using core principles of agent architecture while maintaining full control over the system's behavior.

## What Makes a System "Agentic"?

### Traditional Agent Characteristics:
- **Autonomy**: Self-directed behavior
- **Reactivity**: Responds to environment changes
- **Proactivity**: Goal-oriented behavior
- **Social Ability**: Multi-agent communication
- **Learning**: Improves over time

### Our System's Agentic Qualities:
- **Intelligent Processing**: Multi-strategy decision making
- **Adaptive Behavior**: Context-aware responses
- **Complex Reasoning**: Multi-step problem solving
- **Memory**: Session-based state management
- **Goal Orientation**: Task-specific query handling

## Core Architecture: Pure Python Implementation

### 1. Multi-Strategy Decision Making Engine

Instead of using LangChain's pre-built chains, we implemented custom decision logic:

```python
# Our RAG System - Multi-Strategy Decision Making
class RAGSystem:
    def search_documents(self, query: str) -> List[Dict]:
        # Strategy 1: Semantic Search (40% weight)
        semantic_scores = self._semantic_search(query)
        
        # Strategy 2: Keyword Search (30% weight)
        keyword_scores = self._keyword_search(query)
        
        # Strategy 3: Fuzzy Search (30% weight)
        fuzzy_scores = self._fuzzy_search(query)
        
        # Intelligent score combination
        final_scores = self._combine_scores(
            semantic_scores, keyword_scores, fuzzy_scores
        )
        
        return self._rank_and_return(final_scores)
```

**Why not LangChain?**: LangChain would abstract this decision-making process. Our custom implementation allows fine-grained control over search strategies and scoring algorithms.

### 2. Context-Aware Query Processing

We built a custom query engine that exhibits intelligent behavior:

```python
class QueryEngine:
    def process_query(self, query: str, query_type: str, 
                     relevant_chunks: List[Dict], documents: Dict) -> Dict:
        # Context-aware processing
        if query_type == "Cross-Paper Comparison":
            response = self._handle_comparison_query(query, context, relevant_chunks)
        elif query_type == "Methodology Summary":
            response = self._handle_methodology_query(query, context, documents)
        elif query_type == "Results Extraction":
            response = self._handle_results_query(query, context, relevant_chunks)
        else:
            response = self._handle_general_query(query, context, relevant_chunks)
        
        return self._format_response(response, relevant_chunks)
```

**Agentic Behavior**: The system makes intelligent decisions about how to process different types of queries, adapting its behavior based on context.

### 3. Multi-Level Processing Pipeline

Our system exhibits complex, multi-step reasoning:

```python
# Document Processing Pipeline
class DocumentProcessor:
    def process_pdf(self, pdf_path: str, filename: str) -> Dict:
        # Multi-strategy extraction (intelligent fallback)
        text = self._extract_text_multi_method(pdf_path)
        tables = self._extract_tables_parallel(pdf_path)
        
        # Intelligent chunking decisions
        chunks = self._smart_chunking(text, tables)
        
        # Context-aware metadata extraction
        metadata = self._extract_metadata(text, filename)
        
        return self._compile_document_data(chunks, metadata, tables)
```

**Agentic Quality**: The system makes intelligent decisions about extraction methods, chunking strategies, and metadata handling based on document characteristics.

## Key Differences from LangChain/LangGraph Approach

### 1. **Custom Decision Trees vs. Pre-built Chains**

**LangChain Way:**
```python
# LangChain approach (what we avoided)
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
```

**Our Custom Way:**
```python
# Our custom approach
class QueryEngine:
    def __init__(self):
        self.strategies = {
            'semantic': SemanticSearch(),
            'keyword': KeywordSearch(),
            'fuzzy': FuzzySearch()
        }
    
    def process_query(self, query):
        # Custom multi-strategy processing
        results = {}
        for strategy_name, strategy in self.strategies.items():
            results[strategy_name] = strategy.search(query)
        
        # Custom scoring and combination
        return self._intelligent_combination(results)
```

### 2. **Custom RAG vs. LangChain RAG**

**LangChain RAG:**
- Pre-built retrieval chains
- Limited customization
- Fixed scoring methods
- Standard chunking strategies

**Our Custom RAG:**
- Multi-strategy retrieval (semantic + keyword + fuzzy)
- Custom scoring algorithms
- Adaptive chunking (section-based + sliding window)
- Context-aware result combination

### 3. **State Management**

**LangGraph Way:**
```python
# LangGraph state management
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("generator", generate_node)
```

**Our Way:**
```python
# Custom state management
class ResearchMinerSystem:
    def __init__(self):
        self.document_store = {}
        self.rag_system = RAGSystem()
        self.query_engine = QueryEngine()
        self.session_state = {}
    
    def process_query(self, query, session_id):
        # Custom state management and processing
        context = self.session_state.get(session_id, {})
        results = self.rag_system.search_documents(query)
        response = self.query_engine.process_query(query, results)
        
        # Update session state
        self.session_state[session_id] = {
            'last_query': query,
            'last_results': results,
            'context': context
        }
        return response
```

## Agentic Behaviors We Implemented

### 1. **Intelligent Strategy Selection**

```python
def _combine_scores(self, semantic_scores, keyword_scores, fuzzy_scores):
    """Intelligent score combination based on query characteristics"""
    combined_scores = {}
    
    for chunk_id in semantic_scores:
        # Adaptive weighting based on query type
        semantic_weight = 0.4
        keyword_weight = 0.3
        fuzzy_weight = 0.3
        
        # Adjust weights based on query characteristics
        if self._is_technical_query(self.current_query):
            keyword_weight += 0.1
            fuzzy_weight -= 0.1
        
        combined_scores[chunk_id] = (
            semantic_weight * semantic_scores.get(chunk_id, 0) +
            keyword_weight * keyword_scores.get(chunk_id, 0) +
            fuzzy_weight * fuzzy_scores.get(chunk_id, 0)
        )
    
    return combined_scores
```

### 2. **Context-Aware Response Generation**

```python
def _handle_comparison_query(self, query, context, relevant_chunks):
    """Specialized handling for cross-paper comparison queries"""
    
    # Intelligent document grouping
    papers = self._group_chunks_by_paper(relevant_chunks)
    
    # Context-aware prompt construction
    comparison_prompt = f"""
    You are analyzing multiple research papers. Compare and contrast the following papers based on: {query}
    
    Papers to compare:
    {self._format_papers_for_comparison(papers)}
    
    Provide a structured comparison highlighting:
    1. Key similarities
    2. Important differences
    3. Methodological approaches
    4. Results and conclusions
    """
    
    return self._generate_ai_response(comparison_prompt)
```

### 3. **Adaptive Content Processing**

```python
def _smart_chunking(self, text: str, tables: List[Dict]) -> List[Dict]:
    """Intelligent chunking based on document structure"""
    
    # Try section-based chunking first
    if self._has_clear_sections(text):
        chunks = self._section_based_chunking(text, tables)
    else:
        # Fallback to sliding window
        chunks = self._sliding_window_chunking(text, tables)
    
    # Enhance chunks with table context
    enhanced_chunks = self._integrate_table_context(chunks, tables)
    
    return enhanced_chunks
```

### 4. **Multi-Method Fallback System**

```python
def _extract_text_multi_method(self, pdf_path: str) -> str:
    """Intelligent text extraction with multiple fallback methods"""
    
    try:
        # Primary method: pdfplumber (best for complex layouts)
        text = self._extract_with_pdfplumber(pdf_path)
        if self._is_extraction_successful(text):
            return text
    except Exception as e:
        logging.warning(f"pdfplumber failed: {e}")
    
    try:
        # Fallback method: PyPDF2 (faster, simpler)
        text = self._extract_with_pypdf2(pdf_path)
        if self._is_extraction_successful(text):
            return text
    except Exception as e:
        logging.warning(f"PyPDF2 failed: {e}")
    
    try:
        # Final fallback: OCR (for scanned documents)
        text = self._extract_with_ocr(pdf_path)
        return text
    except Exception as e:
        logging.error(f"All extraction methods failed: {e}")
        return ""
```

## Benefits of Our Custom Approach

### 1. **Full Control**
- Complete visibility into decision-making process
- Ability to fine-tune every aspect of behavior
- Custom optimization for specific use cases

### 2. **Performance Optimization**
- No framework overhead
- Optimized for our specific use case
- Direct control over memory usage and processing

### 3. **Transparency**
- Clear understanding of how decisions are made
- Easy debugging and troubleshooting
- Explainable AI behavior

### 4. **Flexibility**
- Easy to modify behavior for specific requirements
- Can add new strategies without framework limitations
- Custom integration with external systems

## Comparison: Framework vs. Custom Implementation

| Aspect | LangChain/LangGraph | Our Custom Implementation |
|--------|-------------------|---------------------------|
| **Speed** | Framework overhead | Optimized performance |
| **Control** | Limited customization | Full control |
| **Transparency** | Black box behavior | Complete visibility |
| **Flexibility** | Framework constraints | Unlimited flexibility |
| **Maintenance** | Framework updates | Custom maintenance |
| **Learning Curve** | Framework learning | Pure Python skills |

## Implementation Highlights

### 1. **Multi-Strategy RAG System**
- Three parallel search strategies
- Custom scoring algorithms
- Intelligent result combination
- Context-aware weighting

### 2. **Adaptive Query Processing**
- Query type classification
- Context-aware response generation
- Specialized handlers for different query types
- Source attribution and formatting

### 3. **Intelligent Document Processing**
- Multi-method PDF extraction
- Advanced table processing
- Smart chunking strategies
- Content integration

### 4. **State Management**
- Session-based memory
- Context preservation
- Document library management
- Query history tracking

## Technical Implementation Details

### Core Components:
1. **DocumentProcessor**: Handles PDF processing with intelligent fallbacks
2. **RAGSystem**: Multi-strategy search and retrieval
3. **QueryEngine**: Context-aware query processing and response generation
4. **Interfaces**: Web (Streamlit) and CLI for user interaction

### Key Technologies:
- **scikit-learn**: TF-IDF vectorization and similarity calculations
- **OpenAI API**: Language model integration
- **Multiple PDF libraries**: pdfplumber, PyPDF2, tabula-py, camelot
- **Streamlit**: Web interface framework

## Conclusion

Our custom implementation demonstrates that sophisticated agentic-like behavior can be achieved without relying on frameworks like LangChain or LangGraph. By building everything from scratch using pure Python, we achieved:

1. **Better Performance**: No framework overhead
2. **Full Control**: Complete customization capability
3. **Transparency**: Clear understanding of system behavior
4. **Flexibility**: Easy modification and extension

The system exhibits intelligent behavior through:
- Multi-strategy decision making
- Context-aware processing
- Adaptive behavior based on input characteristics
- Intelligent fallback mechanisms

This approach is particularly valuable when you need:
- High performance requirements
- Specific customization needs
- Full transparency in decision-making
- Direct control over system behavior

While frameworks provide convenience, custom implementation offers the flexibility and control needed for sophisticated AI systems that require specific behaviors and optimizations.