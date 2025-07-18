# Class Structure and Implementation Details

## Current System Classes

### 1. **DocumentProcessor Class**
```python
class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 800
        self.chunk_overlap = 150
        self.min_chunk_size = 100
    
    # Main Processing
    def process_pdf(pdf_path, filename) → Dict
    
    # Text Extraction (3 methods)
    def _extract_text_pdfplumber(pdf_path) → str
    def _extract_text_pypdf2(pdf_path) → str
    def _extract_text_ocr(pdf_path) → str
    
    # Table Extraction (3 methods)
    def _extract_tables(pdf_path) → List[Dict]
    def _extract_tables_tabula(pdf_path) → List[Dict]
    def _extract_tables_camelot(pdf_path) → List[Dict]
    def _extract_tables_pdfplumber(pdf_path) → List[Dict]
    
    # Content Processing
    def _combine_text_and_tables(text, tables) → str
    def _clean_text(text) → str
    def _extract_structured_content(text) → Dict
    
    # Smart Chunking
    def _create_chunks(text, filename) → List[Dict]
    def _create_section_based_chunks(text, filename) → List[Dict]
    def _create_sliding_window_chunks(text, filename, start_id) → List[Dict]
    def _split_long_section(section) → List[str]
    
    # Utilities
    def _dataframe_to_text(df, title) → str
    def _get_page_count(pdf_path) → int
    def _determine_processing_method(text) → str
```

### 2. **RAGSystem Class**
```python
class RAGSystem:
    def __init__(self, model_name='tfidf'):
        # Primary TF-IDF (10k features, trigrams)
        self.vectorizer = TfidfVectorizer(...)
        # Keyword TF-IDF (5k features, unigrams)  
        self.keyword_vectorizer = TfidfVectorizer(...)
        self.document_chunks = []
        self.chunk_embeddings = None
        self.keyword_embeddings = None
        self.fitted = False
        self.keyword_fitted = False
    
    # Document Management
    def add_document(document_data) → None
    
    # Multi-Strategy Retrieval
    def retrieve_relevant_chunks(query, top_k=5) → List[Dict]
    def _semantic_search(query, top_k) → List[Dict]
    def _keyword_search(query, top_k) → List[Dict] 
    def _fuzzy_text_search(query, top_k) → List[Dict]
    def _combine_search_results(semantic, keyword, fuzzy, top_k) → List[Dict]
    
    # Utilities
    def _preprocess_text_for_search(text) → str
    def _update_embeddings_matrix() → None
    def _estimate_page_number(chunk, document_data) → int
```

### 3. **QueryEngine Class**
```python
class QueryEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
    
    # Main Processing
    def process_query(query, query_type, relevant_chunks, documents) → Dict
    
    # Query Type Handlers
    def _handle_general_query(query, context, chunks) → str
    def _handle_comparison_query(query, context, chunks) → str
    def _handle_methodology_query(query, context, documents) → str
    def _handle_results_query(query, context, chunks) → str
    def _handle_conclusion_query(query, context, documents) → str
    
    # Utilities
    def _prepare_context(relevant_chunks) → str
    def _format_sources(relevant_chunks) → List[Dict]
    def _classify_query_type(query) → str
```

### 4. **Main App (app.py)**
```python
# Session State Management
st.session_state.documents = {}
st.session_state.rag_system = RAGSystem()
st.session_state.query_engine = QueryEngine()

# UI Components
- File upload interface
- Document processing pipeline  
- Query input and processing
- Results display with sources
- Document management sidebar
```

## Current Implementation Status

### ✅ **Fully Implemented & Working**
1. **Document Processing**
   - Multi-method PDF text extraction (pdfplumber → PyPDF2 → OCR)
   - Advanced table extraction (tabula + camelot + pdfplumber)
   - Smart chunking (section-based + sliding window)
   - Text cleaning and structuring

2. **RAG System**
   - Multi-strategy search (semantic + keyword + fuzzy)
   - Enhanced TF-IDF with trigrams and large vocabulary
   - Weighted score combination
   - Debug information for search methods

3. **Query Engine**
   - OpenAI GPT-4o integration
   - Multiple query type handlers
   - Context preparation and source attribution
   - Structured response formatting

4. **User Interface**
   - Streamlit web application
   - File upload and management
   - Query interface with type selection
   - Results display with source details
   - Document statistics and debugging info

### 🔧 **Configuration Parameters**

#### Text Processing
```python
chunk_size = 800           # Characters per chunk
chunk_overlap = 150        # Overlap between chunks  
min_chunk_size = 100       # Minimum viable chunk
```

#### TF-IDF Settings
```python
# Semantic TF-IDF
max_features = 10000       # Vocabulary size
ngram_range = (1, 3)       # Unigrams to trigrams
max_df = 0.9              # Document frequency filter
min_df = 1                # Minimum document frequency

# Keyword TF-IDF  
max_features = 5000        # Smaller vocabulary
ngram_range = (1, 1)       # Unigrams only
stop_words = None          # Keep all words
```

#### Search Weights
```python
semantic_weight = 0.4      # 40% semantic similarity
keyword_weight = 0.3       # 30% keyword matching
fuzzy_weight = 0.3         # 30% fuzzy text matching
```

## Static vs Dynamic Functionality

### **Static Components** (Work Offline)
- PDF text extraction and cleaning
- Table detection and extraction
- Text chunking and preprocessing
- TF-IDF vectorization and indexing
- Multi-strategy search algorithms
- Document management and storage
- UI interface and navigation

### **Dynamic Components** (Require API Key)
- OpenAI GPT-4o response generation
- AI-powered query understanding
- Intelligent answer synthesis
- Context-aware responses

The core document processing and search functionality is completely self-contained and works without internet connection. Only the final AI response generation requires the OpenAI API.