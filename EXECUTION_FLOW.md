# Query Execution Flow - Step by Step

## Complete Flow Diagram

```
USER UPLOADS PDF
       ↓
┌─────────────────────────────────────────────────────────────┐
│                  DOCUMENT PROCESSING                        │
├─────────────────────────────────────────────────────────────┤
│ DocumentProcessor.process_pdf()                             │
│ ├── _extract_text_pdfplumber() → Primary text extraction    │
│ ├── _extract_text_pypdf2() → Fallback if needed            │
│ ├── _extract_text_ocr() → OCR for scanned PDFs             │
│ ├── _extract_tables() → Multi-method table extraction      │
│ │   ├── _extract_tables_tabula() → Java-based              │
│ │   ├── _extract_tables_camelot() → Advanced detection     │
│ │   └── _extract_tables_pdfplumber() → Native Python       │
│ ├── _combine_text_and_tables() → Merge content             │
│ ├── _clean_text() → Text normalization                     │
│ ├── _extract_structured_content() → Get title, sections    │
│ └── _create_chunks() → Smart chunking                      │
│     ├── _create_section_based_chunks() → Natural sections  │
│     └── _create_sliding_window_chunks() → Overlapping      │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    RAG INDEXING                            │
├─────────────────────────────────────────────────────────────┤
│ RAGSystem.add_document()                                    │
│ ├── Extract chunk texts                                     │
│ ├── Fit TF-IDF vectorizers (if first document)             │
│ │   ├── Primary: 10k features, trigrams, L2 norm           │
│ │   └── Keyword: 5k features, unigrams only                │
│ ├── Generate embeddings for all chunks                     │
│ │   ├── Semantic embeddings (primary vectorizer)           │
│ │   └── Keyword embeddings (keyword vectorizer)            │
│ ├── Store enhanced chunks with metadata                    │
│ └── Update search indices                                  │
└─────────────────────────────────────────────────────────────┘
       ↓
    DOCUMENT READY FOR QUERIES
       ↓
USER SUBMITS QUERY
       ↓
┌─────────────────────────────────────────────────────────────┐
│                   RETRIEVAL PHASE                          │
├─────────────────────────────────────────────────────────────┤
│ RAGSystem.retrieve_relevant_chunks()                        │
│ │                                                          │
│ ├── STRATEGY 1: Semantic Search (40% weight)               │
│ │   ├── Transform query with primary TF-IDF                │
│ │   ├── Calculate cosine similarity with chunk embeddings  │
│ │   └── Get top results with semantic scores               │
│ │                                                          │
│ ├── STRATEGY 2: Keyword Search (30% weight)                │
│ │   ├── Transform query with keyword TF-IDF                │
│ │   ├── Calculate similarity with keyword embeddings       │
│ │   └── Get top results with keyword scores                │
│ │                                                          │
│ ├── STRATEGY 3: Fuzzy Text Search (30% weight)             │
│ │   ├── Extract words from query                           │
│ │   ├── Calculate Jaccard similarity with chunk words      │
│ │   ├── Apply phrase boost for exact matches               │
│ │   └── Get top results with fuzzy scores                  │
│ │                                                          │
│ └── _combine_search_results()                              │
│     ├── Aggregate scores from all strategies               │
│     ├── Apply weighted combination                         │
│     ├── Sort by total score                                │
│     └── Return top-k chunks with debug info                │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                  RESPONSE GENERATION                        │
├─────────────────────────────────────────────────────────────┤
│ QueryEngine.process_query()                                 │
│ ├── _prepare_context() → Format relevant chunks            │
│ ├── Determine query type and select handler:               │
│ │   ├── _handle_general_query() → Default processing       │
│ │   ├── _handle_comparison_query() → Cross-paper analysis  │
│ │   ├── _handle_methodology_query() → Method extraction    │
│ │   ├── _handle_results_query() → Results extraction       │
│ │   └── _handle_conclusion_query() → Conclusion summary    │
│ ├── Call OpenAI GPT-4o with prepared context              │
│ ├── _format_sources() → Create source attribution         │
│ └── Return structured response                             │
└─────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────┐
│                    UI DISPLAY                              │
├─────────────────────────────────────────────────────────────┤
│ Streamlit App Display                                       │
│ ├── Show AI-generated answer                               │
│ ├── Display sources with metadata:                         │
│ │   ├── Document name and page                             │
│ │   ├── Relevance score                                    │
│ │   ├── Search methods used                                │
│ │   └── Content preview (300 chars)                        │
│ ├── Update document statistics                             │
│ └── Show debug information (if enabled)                    │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow During Query Execution

### 1. Input Processing
```python
# User query: "What methodology was used in the study?"
query = user_input
query_type = "auto-detected"  # or user-selected
```

### 2. Retrieval Results
```python
# RAG returns structured chunks
relevant_chunks = [
    {
        'id': 'paper1_section_3',
        'text': 'methodology content...',
        'similarity_score': 0.85,
        'search_methods': ['semantic: 0.78', 'keyword: 0.65', 'fuzzy: 0.42'],
        'document': 'research_paper.pdf',
        'page': 3
    },
    # ... more chunks
]
```

### 3. Context Preparation
```python
# QueryEngine formats context
context = """
Based on the following research paper content:

[Source 1 - research_paper.pdf, Page 3]
methodology content...

[Source 2 - research_paper.pdf, Page 4]  
additional methodology details...
"""
```

### 4. AI Response Generation
```python
# OpenAI GPT-4o processes the query
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "research paper analysis prompt"},
        {"role": "user", "content": f"Query: {query}\nContext: {context}"}
    ]
)
```

### 5. Final Output Structure
```python
{
    'answer': 'The study used a mixed-methods approach...',
    'sources': [
        {
            'document': 'research_paper.pdf',
            'page': 3,
            'score': 0.85,
            'search_methods': ['semantic: 0.78', 'keyword: 0.65'],
            'content': 'methodology content preview...'
        }
    ],
    'query_type': 'Methodology Summary',
    'num_sources': 3
}
```

## Class Interaction Summary

```
app.py (UI Layer)
    ↓ calls
DocumentProcessor (Processing Layer)  
    ↓ stores results in
RAGSystem (Retrieval Layer)
    ↓ provides chunks to  
QueryEngine (AI Layer)
    ↓ returns response to
app.py (UI Display)
```

## Static Components (Work Without Internet)

1. **PDF Processing**: Text and table extraction
2. **Text Chunking**: Section-based and sliding window
3. **TF-IDF Indexing**: Document vectorization  
4. **Search Algorithms**: Semantic, keyword, and fuzzy search
5. **UI Interface**: File upload and document management

## Dynamic Components (Require OpenAI API)

1. **Response Generation**: GPT-4o integration
2. **Query Understanding**: AI-powered analysis
3. **Source Attribution**: Intelligent citation

The system can process documents and perform search operations entirely offline, only requiring internet connection for AI response generation.