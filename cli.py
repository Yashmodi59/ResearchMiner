#!/usr/bin/env python3
"""
CLI interface for AI Research Paper Analysis Agent
Usage: python cli.py [command] [options]
"""

import argparse
import os
import sys
import json
from pathlib import Path
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from query_engine import QueryEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResearchPaperCLI:
    """Command-line interface for research paper analysis"""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.rag_system = RAGSystem()
        self.query_engine = QueryEngine()
        self.documents = {}
        
    def process_document(self, pdf_path: str) -> bool:
        """Process a single PDF document"""
        if not os.path.exists(pdf_path):
            print(f"Error: File {pdf_path} not found")
            return False
            
        filename = os.path.basename(pdf_path)
        print(f"Processing {filename}...")
        
        try:
            document_data = self.doc_processor.process_pdf(pdf_path, filename)
            
            if document_data:
                self.documents[filename] = document_data
                self.rag_system.add_document(document_data)
                
                # Print processing summary
                print(f"✓ Successfully processed {filename}")
                print(f"  - Pages: {document_data['total_pages']}")
                print(f"  - Text chunks: {len(document_data['chunks'])}")
                print(f"  - Tables found: {len(document_data.get('tables', []))}")
                print(f"  - Characters: {len(document_data['full_text']):,}")
                print(f"  - Method: {document_data['processing_method']}")
                
                return True
            else:
                print(f"✗ Failed to process {filename}")
                return False
                
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            return False
    
    def process_directory(self, directory_path: str) -> int:
        """Process all PDF files in a directory"""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            print(f"Error: Directory {directory_path} not found")
            return 0
            
        pdf_files = list(directory.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return 0
            
        successful = 0
        print(f"Found {len(pdf_files)} PDF files to process...")
        
        for pdf_file in pdf_files:
            if self.process_document(str(pdf_file)):
                successful += 1
                
        print(f"\nProcessed {successful}/{len(pdf_files)} documents successfully")
        return successful
    
    def query_documents(self, query: str, query_type: str = "auto", top_k: int = 5) -> dict:
        """Query the processed documents"""
        if not self.documents:
            print("Error: No documents loaded. Please process documents first.")
            return {}
            
        print(f"Searching for: '{query}'")
        print("Retrieving relevant content...")
        
        try:
            # Get relevant chunks
            relevant_chunks = self.rag_system.retrieve_relevant_chunks(query, top_k=top_k)
            
            if not relevant_chunks:
                print("No relevant content found for your query.")
                return {}
            
            print(f"Found {len(relevant_chunks)} relevant chunks")
            
            # Generate response
            response = self.query_engine.process_query(
                query=query,
                query_type=query_type,
                relevant_chunks=relevant_chunks,
                documents=self.documents
            )
            
            return response
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {}
    
    def display_response(self, response: dict, show_sources: bool = True, show_debug: bool = False):
        """Display query response in formatted way"""
        if not response:
            return
            
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(response.get('answer', 'No answer generated'))
        
        if show_sources and response.get('sources'):
            print("\n" + "-"*80)
            print("SOURCES:")
            print("-"*80)
            
            for i, source in enumerate(response['sources'], 1):
                print(f"\n[{i}] {source['document']} (Page {source.get('page', 'N/A')})")
                print(f"    Relevance Score: {source.get('score', 0):.3f}")
                
                if show_debug and 'search_methods' in source:
                    print(f"    Search Methods: {', '.join(source['search_methods'])}")
                
                # Show content preview
                content = source['content']
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"    Preview: {preview}")
        
        print("\n" + "="*80)
    
    def list_documents(self):
        """List all processed documents"""
        if not self.documents:
            print("No documents loaded.")
            return
            
        print(f"Loaded Documents ({len(self.documents)}):")
        print("-" * 50)
        
        for filename, doc_data in self.documents.items():
            print(f"• {filename}")
            print(f"  Pages: {doc_data['total_pages']}, "
                  f"Chunks: {len(doc_data['chunks'])}, "
                  f"Tables: {len(doc_data.get('tables', []))}")
    
    def save_session(self, filepath: str):
        """Save current session data"""
        try:
            session_data = {
                'documents': {name: {
                    'filename': doc['filename'],
                    'total_pages': doc['total_pages'],
                    'processing_method': doc['processing_method'],
                    'chunk_count': len(doc['chunks']),
                    'table_count': len(doc.get('tables', []))
                } for name, doc in self.documents.items()}
            }
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
                
            print(f"Session saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving session: {str(e)}")
    
    def interactive_mode(self):
        """Start interactive query mode"""
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        print("Type your questions below. Use 'quit' or 'exit' to stop.")
        print("Commands:")
        print("  'list' - Show loaded documents")
        print("  'help' - Show this help")
        print("-" * 60)
        
        while True:
            try:
                query = input("\nQuery> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                elif query.lower() == 'list':
                    self.list_documents()
                elif query.lower() == 'help':
                    print("Available commands:")
                    print("  list - Show loaded documents")
                    print("  help - Show this help")
                    print("  quit/exit/q - Exit interactive mode")
                elif query:
                    response = self.query_documents(query)
                    self.display_response(response)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

def main():
    parser = argparse.ArgumentParser(
        description="AI Research Paper Analysis Agent - CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single document
  python cli.py process paper.pdf
  
  # Process all PDFs in a directory
  python cli.py process-dir /path/to/papers/
  
  # Query documents
  python cli.py query "What methodology was used?"
  
  # Interactive mode
  python cli.py interactive
  
  # Process and then query
  python cli.py process paper.pdf --query "What are the main findings?"
        """
    )
    
    parser.add_argument('command', choices=['process', 'process-dir', 'query', 'interactive', 'list'],
                      help='Command to execute')
    parser.add_argument('input', nargs='?', help='Input file, directory, or query text')
    parser.add_argument('--query', '-q', help='Query to run after processing')
    parser.add_argument('--query-type', choices=['auto', 'Direct Content Lookup', 'Cross-Paper Comparison', 
                                               'Methodology Summary', 'Results Extraction', 'Conclusion Summary'],
                      default='auto', help='Type of query to perform')
    parser.add_argument('--top-k', type=int, default=5, help='Number of relevant chunks to retrieve')
    parser.add_argument('--show-sources', action='store_true', default=True, help='Show source information')
    parser.add_argument('--show-debug', action='store_true', help='Show debug information')
    parser.add_argument('--save-session', help='Save session data to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize CLI
    cli = ResearchPaperCLI()
    
    # Execute command
    if args.command == 'process':
        if not args.input:
            print("Error: Please provide a PDF file path")
            sys.exit(1)
        
        success = cli.process_document(args.input)
        
        # Run query if provided
        if success and args.query:
            response = cli.query_documents(args.query, args.query_type, args.top_k)
            cli.display_response(response, args.show_sources, args.show_debug)
            
    elif args.command == 'process-dir':
        if not args.input:
            print("Error: Please provide a directory path")
            sys.exit(1)
        
        count = cli.process_directory(args.input)
        
        # Run query if provided
        if count > 0 and args.query:
            response = cli.query_documents(args.query, args.query_type, args.top_k)
            cli.display_response(response, args.show_sources, args.show_debug)
            
    elif args.command == 'query':
        if not args.input:
            print("Error: Please provide a query")
            sys.exit(1)
        
        response = cli.query_documents(args.input, args.query_type, args.top_k)
        cli.display_response(response, args.show_sources, args.show_debug)
        
    elif args.command == 'list':
        cli.list_documents()
        
    elif args.command == 'interactive':
        cli.interactive_mode()
    
    # Save session if requested
    if args.save_session:
        cli.save_session(args.save_session)

if __name__ == "__main__":
    main()