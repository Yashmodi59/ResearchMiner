# AI Research Paper Analysis System - Code Architecture Overview

## System Architecture

The system consists of 5 main classes with clear separation of concerns:

```
📄 app.py (Main UI)
├── 🔧 DocumentProcessor (document_processor.py)
├── 🔍 RAGSystem (rag_system.py)
├── 🤖 QueryEngine (query_engine.py)
└── 🛠️ Utils (utils.py)
```

## Class Breakdown

### 1. **DocumentProcessor** (`document_processor.py`)
**Purpose**: Handles PDF processing and text extraction

**Key Methods**:
- `process_pdf(pdf_path, filename)` - Main processing entry point
- `_extract_text_pdfplumber(pdf_path)` - Primary text extraction
- `_extract_text_pypdf2(pdf_path)` - Fallback text extraction
- `_extract_tables(pdf_path)` - Multi-method table extraction
- `_create_chunks(text, filename)` - Smart text chunking
- `_create_section_based_chunks()` - Section-aware chunking
- `_create_sliding_window_chunks()` - Overlapping chunks

**Table Extraction Methods**:
- `_extract_tables_tabula()` - Java-based table extraction
- `_extract_tables_camelot()` - Advanced table detection
- `_extract_tables_pdfplumber()` - Native Python tables

### 2. **RAGSystem** (`rag_system.py`)
**Purpose**: Semantic search and document retrieval

**Key Methods**:
- `add_document(document_data)` - Add processed documents
- `retrieve_relevant_chunks(query, top_k)` - Multi-strategy search
- `_semantic_search(query, top_k)` - TF-IDF semantic search
- `_keyword_search(query, top_k)` - Exact keyword matching
- `_fuzzy_text_search(query, top_k)` - Phrase overlap detection
- `_combine_search_results()` - Weighted score combination

**Search Configuration**:
- **Primary TF-IDF**: 10,000 features, trigrams, L2 normalization
- **Keyword TF-IDF**: 5,000 features, unigrams only
- **Fuzzy Search**: Jaccard similarity + phrase boosting

### 3. **QueryEngine** (`query_engine.py`)
**Purpose**: Processes queries and generates AI responses

**Key Methods**:
- `process_query(query, query_type, chunks, documents)` - Main processing
- `_handle_general_query()` - Default query handling
- `_handle_comparison_query()` - Cross-paper comparisons
- `_handle_methodology_query()` - Methodology extraction
- `_handle_results_query()` - Results extraction
- `_handle_conclusion_query()` - Conclusion summaries

**AI Configuration**:
- Model: GPT-4o (latest OpenAI model)
- Context preparation from relevant chunks
- Source attribution and citations

### 4. **Main App** (`app.py`)
**Purpose**: Streamlit UI and orchestration

**Key Components**:
- Session state management
- File upload interface
- Document processing pipeline
- Query interface
- Results display with debugging info

## Query Execution Flow

### Step 1: Document Upload & Processing
```
User uploads PDF → DocumentProcessor.process_pdf()
├── Text extraction (pdfplumber → PyPDF2 → OCR)
├── Table extraction (tabula + camelot + pdfplumber)
├── Text cleaning and structuring
├── Smart chunking (section-based + sliding window)
└── Store in session state
```

### Step 2: Document Indexing
```
RAGSystem.add_document()
├── Extract text from chunks
├── Fit TF-IDF vectorizers (semantic + keyword)
├── Generate embeddings for all chunks
├── Store chunks with metadata
└── Update search indices
```

### Step 3: Query Processing
```
User submits query → RAGSystem.retrieve_relevant_chunks()
├── Strategy 1: Semantic TF-IDF search (weight: 0.4)
├── Strategy 2: Keyword exact matching (weight: 0.3)
├── Strategy 3: Fuzzy text matching (weight: 0.3)
├── Combine and rank results
└── Return top-k chunks with scores
```

### Step 4: Response Generation
```
QueryEngine.process_query()
├── Prepare context from relevant chunks
├── Determine query type and handler
├── Generate AI response using GPT-4o
├── Format sources with attribution
└── Return structured response
```

### Step 5: UI Display
```
Streamlit displays results
├── Show AI-generated answer
├── Display sources with scores
├── Show search method details
└── Update document statistics
```

## Data Structures

### Document Data
```python
{
    'filename': str,
    'full_text': str,
    'structured_content': dict,
    'tables': list,
    'chunks': list,
    'total_pages': int,
    'processing_method': str
}
```

### Chunk Data
```python
{
    'id': str,
    'text': str,
    'document': str,
    'start_pos': int,
    'end_pos': int,
    'chunk_id': int,
    'chunk_type': str,
    'embedding': array,
    'keyword_embedding': array,
    'preprocessed_text': str
}
```

### Table Data
```python
{
    'source': str,
    'table_id': str,
    'content': str,
    'dataframe': DataFrame,
    'accuracy': float (optional),
    'page': int (optional)
}
```

## Search Strategy Weights

The system uses weighted combination of three search methods:
- **Semantic Search**: 40% weight (TF-IDF with trigrams)
- **Keyword Search**: 30% weight (exact term matching)
- **Fuzzy Search**: 30% weight (word overlap + phrase detection)

## Configuration Parameters

### Text Processing
- Chunk size: 800 characters
- Chunk overlap: 150 characters
- Minimum chunk size: 100 characters

### TF-IDF Settings
- Max features: 10,000 (semantic), 5,000 (keyword)
- N-gram range: (1,3) semantic, (1,1) keyword
- Min/Max document frequency: 1/0.9

### Query Processing
- Top-k chunks retrieved: 8
- Search methods: 3 parallel strategies
- Response model: GPT-4o

## Static vs Dynamic Components

### Static Components (Always Work)
- Document upload interface
- File processing pipeline
- Text extraction methods
- Chunking algorithms
- TF-IDF vectorization
- UI layout and navigation

### Dynamic Components (Require API)
- OpenAI response generation
- Real-time query processing
- AI-powered analysis

The core RAG functionality works entirely offline using scikit-learn and doesn't require external APIs for document processing and search.