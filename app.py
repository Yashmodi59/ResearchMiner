import streamlit as st
import os
import tempfile
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from query_engine import QueryEngine
import json

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = RAGSystem()
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = QueryEngine()

st.set_page_config(
    page_title="ResearchMiner - AI Research Paper Analysis",
    page_icon="📄",
    layout="wide"
)

st.title("📄 ResearchMiner - AI Research Paper Analysis")
st.markdown("Upload research papers and ask complex questions using advanced RAG capabilities")

# Sidebar for document management
with st.sidebar:
    st.header("📚 Document Management")
    
    # Upload section
    uploaded_files = st.file_uploader(
        "Upload Research Papers (PDF)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF research papers for analysis"
    )
    
    # Process uploaded files
    if uploaded_files:
        doc_processor = DocumentProcessor()
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.documents:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Process document
                        document_data = doc_processor.process_pdf(tmp_path, uploaded_file.name)
                        
                        if document_data:
                            # Store document
                            st.session_state.documents[uploaded_file.name] = document_data
                            
                            # Add to RAG system
                            st.session_state.rag_system.add_document(document_data)
                            
                            st.success(f"✅ {uploaded_file.name} processed successfully")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")
                        
                        # Clean up temporary file
                        os.unlink(tmp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
    
    # Display processed documents
    if st.session_state.documents:
        st.subheader("📋 Processed Documents")
        for doc_name, doc_data in st.session_state.documents.items():
            with st.expander(f"📄 {doc_name}"):
                st.write(f"**Pages:** {doc_data.get('total_pages', 'N/A')}")
                st.write(f"**Chunks:** {len(doc_data.get('chunks', []))}")
                st.write(f"**Size:** {len(doc_data.get('full_text', ''))} characters")
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents", type="secondary"):
            st.session_state.documents = {}
            st.session_state.rag_system = RAGSystem()
            st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Query Interface")
    
    # Query input
    query = st.text_area(
        "Ask a question about your research papers:",
        placeholder="Examples:\n• What is the conclusion of Paper X?\n• Compare the results of Paper A and Paper B\n• Summarize the methodology of Paper C\n• What are the evaluation metrics used?",
        height=100
    )
    
    # Query type selection with auto-detection option
    query_type_options = [
        "🤖 Auto-Detect (Recommended)",
        "General Question",
        "Direct Content Lookup",
        "Cross-Paper Comparison", 
        "Methodology Summary",
        "Results Extraction",
        "Conclusion Summary"
    ]
    
    selected_query_type = st.selectbox(
        "Query Type:",
        query_type_options,
        help="Choose 'Auto-Detect' to let AI determine the best analysis type, or manually select a specific type"
    )
    
    # Convert selection to internal format
    if selected_query_type == "🤖 Auto-Detect (Recommended)":
        query_type = None  # Triggers auto-detection
        st.info("💡 **Smart Mode**: AI will automatically determine the best analysis approach based on your question")
    else:
        query_type = selected_query_type
        st.info(f"🎯 **Manual Mode**: Will use {selected_query_type} analysis")
    
    # Process query
    if st.button("🚀 Ask Question", type="primary", disabled=not query or not st.session_state.documents):
        if not st.session_state.documents:
            st.warning("⚠️ Please upload some research papers first")
        elif not query.strip():
            st.warning("⚠️ Please enter a question")
        else:
            with st.spinner("🤔 Analyzing documents and generating response..."):
                try:
                    # Get relevant chunks using enhanced RAG
                    relevant_chunks = st.session_state.rag_system.retrieve_relevant_chunks(query, top_k=8)
                    
                    # Generate response using query engine
                    response = st.session_state.query_engine.process_query(
                        query=query,
                        query_type=query_type,  # None for auto-detect, or specific type
                        relevant_chunks=relevant_chunks,
                        documents=st.session_state.documents
                    )
                    
                    # Display response with detected query type
                    st.subheader("💡 Response")
                    
                    # Show query type (detected or manual)
                    final_query_type = response.get('query_type', 'General Question')
                    if selected_query_type == "🤖 Auto-Detect (Recommended)":
                        st.caption(f"🤖 **AI Detected**: {final_query_type}")
                    else:
                        st.caption(f"🎯 **Manual Selection**: {final_query_type}")
                    
                    st.write(response['answer'])
                    
                    # Display sources with enhanced debugging info
                    if response.get('sources'):
                        with st.expander("📚 Sources & Search Details"):
                            for i, source in enumerate(response['sources'], 1):
                                st.write(f"**Source {i}:** {source['document']} (Page {source.get('page', 'N/A')})")
                                st.write(f"*Relevance Score: {source.get('score', 0):.3f}*")
                                if 'search_methods' in source:
                                    st.write(f"*Search Methods: {', '.join(source['search_methods'])}*")
                                st.write(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
                                st.divider()
                    
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")

with col2:
    st.header("📊 Analysis Stats")
    
    if st.session_state.documents:
        total_docs = len(st.session_state.documents)
        total_chunks = sum(len(doc.get('chunks', [])) for doc in st.session_state.documents.values())
        total_chars = sum(len(doc.get('full_text', '')) for doc in st.session_state.documents.values())
        
        st.metric("Documents", total_docs)
        st.metric("Text Chunks", total_chunks)
        st.metric("Total Characters", f"{total_chars:,}")
        
        # Document breakdown
        st.subheader("📄 Document Breakdown")
        for doc_name, doc_data in st.session_state.documents.items():
            chunks = len(doc_data.get('chunks', []))
            tables = len(doc_data.get('tables', []))
            st.write(f"**{doc_name}:** {chunks} chunks, {tables} tables")
    else:
        st.info("Upload documents to see analysis statistics")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>AI Research Paper Analysis Agent • Powered by OpenAI GPT-4o • Custom RAG Implementation</p>
    </div>
    """,
    unsafe_allow_html=True
)
