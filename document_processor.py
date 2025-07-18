import PyPDF2
import pdfplumber
import pandas as pd
from PIL import Image
import io
import re
from typing import Dict, List, Optional
import logging
import warnings

# Suppress common PDF processing warnings that don't affect functionality
warnings.filterwarnings("ignore", message=".*Cannot set gray level.*")
warnings.filterwarnings("ignore", message=".*Failed to import jpype.*")
logging.getLogger('pdfminer.pdfinterp').setLevel(logging.ERROR)
logging.getLogger('tabula.backend').setLevel(logging.ERROR)

# Try to import optional table extraction libraries
try:
    # Try multiple import patterns for tabula-py
    try:
        import tabula.io as tabula_io
        tabula_read_pdf = tabula_io.read_pdf
        TABULA_AVAILABLE = True
    except:
        try:
            from tabula.io import read_pdf as tabula_read_pdf
            TABULA_AVAILABLE = True
        except:
            import tabula
            tabula_read_pdf = tabula.read_pdf
            TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    logging.info("tabula-py not available - using pdfplumber for table extraction")

try:
    import camelot
    # Test if camelot has the expected functionality
    if hasattr(camelot, 'read_pdf'):
        CAMELOT_AVAILABLE = True
    else:
        CAMELOT_AVAILABLE = False
        logging.info("Camelot version incompatible - using pdfplumber for table extraction")
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.info("Camelot not available - using pdfplumber for table extraction")

class DocumentProcessor:
    """Handles PDF processing, text extraction, and OCR"""
    
    def __init__(self):
        self.chunk_size = 800   # Smaller chunks for better accuracy
        self.chunk_overlap = 150  # Overlap between chunks
        self.min_chunk_size = 100  # Minimum viable chunk size
        
    def process_pdf(self, pdf_path: str, filename: str) -> Optional[Dict]:
        """
        Process a PDF file and extract structured content
        
        Args:
            pdf_path: Path to the PDF file
            filename: Original filename
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Extract text content using multi-method approach
            text_content = self._extract_text_pdfplumber(pdf_path)
            
            # If pdfplumber fails or returns minimal text, try PyPDF2
            if not text_content or len(text_content.strip()) < 100:
                text_content = self._extract_text_pypdf2(pdf_path)
            
            # If still minimal text, use OCR
            if not text_content or len(text_content.strip()) < 100:
                text_content = self._extract_text_ocr(pdf_path)
            
            # Extract tables separately using specialized tools
            tables_content = self._extract_tables(pdf_path)
            
            if not text_content:
                return None
            
            # Combine text content with table content
            combined_content = self._combine_text_and_tables(text_content, tables_content)
                
            # Clean and structure the text
            cleaned_text = self._clean_text(combined_content)
            
            # Extract structured information
            structured_content = self._extract_structured_content(cleaned_text)
            
            # Create chunks for RAG
            chunks = self._create_chunks(cleaned_text, filename)
            
            # Get page count
            page_count = self._get_page_count(pdf_path)
            
            return {
                'filename': filename,
                'full_text': cleaned_text,
                'structured_content': structured_content,
                'tables': tables_content,
                'chunks': chunks,
                'total_pages': page_count,
                'processing_method': self._determine_processing_method(text_content)
            }
            
        except Exception as e:
            logging.error(f"Error processing PDF {filename}: {str(e)}")
            return None
    
    def _extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logging.warning(f"pdfplumber extraction failed: {str(e)}")
            return ""
    
    def _extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logging.warning(f"PyPDF2 extraction failed: {str(e)}")
            return ""
    
    def _extract_text_ocr(self, pdf_path: str) -> str:
        """Extract text using OCR for scanned PDFs"""
        try:
            # For now, return empty string since OCR requires additional setup
            # This can be enhanced later with proper OCR implementation
            logging.info("OCR extraction not implemented yet - falling back to text-based extraction")
            return ""
            
        except Exception as e:
            logging.warning(f"OCR extraction failed: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (basic patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        # Clean up common PDF artifacts
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\`\~\@\#\$\%\^\&\*\+\=\|\\\/\<\>]', ' ', text)
        
        return text.strip()
    
    def _extract_structured_content(self, text: str) -> Dict:
        """Extract structured information from the text"""
        structured = {
            'title': '',
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'conclusion': '',
            'references': ''
        }
        
        # Simple regex patterns for common paper sections
        patterns = {
            'abstract': r'(?i)abstract\s*[:\-]?\s*(.*?)(?=introduction|1\.|methodology|related work|\n\n\w+:)',
            'introduction': r'(?i)introduction\s*[:\-]?\s*(.*?)(?=methodology|related work|background|literature review|\n\n\w+:)',
            'methodology': r'(?i)(?:methodology|methods?|approach)\s*[:\-]?\s*(.*?)(?=results|experiments?|evaluation|discussion|\n\n\w+:)',
            'results': r'(?i)(?:results?|experiments?|evaluation)\s*[:\-]?\s*(.*?)(?=discussion|conclusion|references|\n\n\w+:)',
            'conclusion': r'(?i)(?:conclusion|conclusions?|summary)\s*[:\-]?\s*(.*?)(?=references|acknowledgment|\n\n\w+:)',
            'references': r'(?i)references?\s*[:\-]?\s*(.*?)$'
        }
        
        for section, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL)
            if match:
                structured[section] = match.group(1).strip()[:2000]  # Limit length
        
        # Extract title (usually first significant text block)
        title_match = re.search(r'^(.{10,100})\n', text)
        if title_match:
            structured['title'] = title_match.group(1).strip()
        
        return structured
    
    def _create_chunks(self, text: str, filename: str) -> List[Dict]:
        """Create intelligently chunked pieces for RAG processing"""
        chunks = []
        
        if not text:
            return chunks
        
        # First try to split by sections (paragraphs)
        section_chunks = self._create_section_based_chunks(text, filename)
        if section_chunks:
            chunks.extend(section_chunks)
        
        # Also create overlapping sliding window chunks for comprehensive coverage
        sliding_chunks = self._create_sliding_window_chunks(text, filename, len(section_chunks))
        chunks.extend(sliding_chunks)
        
        return chunks
    
    def _create_section_based_chunks(self, text: str, filename: str) -> List[Dict]:
        """Create chunks based on natural text sections"""
        chunks = []
        
        # Split by double newlines (paragraphs) or section headers
        sections = re.split(r'\n\s*\n|\n(?=[A-Z][^a-z]*\n)', text)
        
        chunk_id = 0
        current_pos = 0
        
        for section in sections:
            section = section.strip()
            if len(section) < self.min_chunk_size:
                continue
                
            # If section is too long, split it further
            if len(section) > self.chunk_size * 1.5:
                subsections = self._split_long_section(section)
                for subsection in subsections:
                    if len(subsection.strip()) >= self.min_chunk_size:
                        chunks.append({
                            'id': f"{filename}_section_{chunk_id}",
                            'text': subsection.strip(),
                            'document': filename,
                            'start_pos': current_pos,
                            'end_pos': current_pos + len(subsection),
                            'chunk_id': chunk_id,
                            'chunk_type': 'section'
                        })
                        chunk_id += 1
                        current_pos += len(subsection)
            else:
                chunks.append({
                    'id': f"{filename}_section_{chunk_id}",
                    'text': section,
                    'document': filename,
                    'start_pos': current_pos,
                    'end_pos': current_pos + len(section),
                    'chunk_id': chunk_id,
                    'chunk_type': 'section'
                })
                chunk_id += 1
                current_pos += len(section)
        
        return chunks
    
    def _create_sliding_window_chunks(self, text: str, filename: str, start_id: int) -> List[Dict]:
        """Create overlapping sliding window chunks"""
        chunks = []
        start = 0
        chunk_id = start_id
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n\n']:
                    sentence_end = text.find(punct, end - 100, end + 100)
                    if sentence_end != -1:
                        end = sentence_end + len(punct)
                        break
            
            chunk_text = text[start:end].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'id': f"{filename}_sliding_{chunk_id}",
                    'text': chunk_text,
                    'document': filename,
                    'start_pos': start,
                    'end_pos': end,
                    'chunk_id': chunk_id,
                    'chunk_type': 'sliding'
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            if start >= len(text) - self.min_chunk_size:
                break
        
        return chunks
    
    def _split_long_section(self, section: str) -> List[str]:
        """Split a long section into smaller manageable pieces"""
        # Try to split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', section)
        
        subsections = []
        current_subsection = ""
        
        for sentence in sentences:
            if len(current_subsection + sentence) <= self.chunk_size:
                current_subsection += sentence + " "
            else:
                if current_subsection.strip():
                    subsections.append(current_subsection.strip())
                current_subsection = sentence + " "
        
        # Add the last subsection
        if current_subsection.strip():
            subsections.append(current_subsection.strip())
        
        return subsections
    
    def _extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF using multiple methods"""
        tables = []
        
        try:
            # Method 1: Use tabula-py for table extraction (if available and Java present)
            if TABULA_AVAILABLE:
                tabula_tables = self._extract_tables_tabula(pdf_path)
                tables.extend(tabula_tables)
            
            # Method 2: Use camelot for better table detection (if available)
            if CAMELOT_AVAILABLE:
                camelot_tables = self._extract_tables_camelot(pdf_path)
                tables.extend(camelot_tables)
            
            # Method 3: Use pdfplumber for table extraction (always available)
            pdfplumber_tables = self._extract_tables_pdfplumber(pdf_path)
            tables.extend(pdfplumber_tables)
            
        except Exception as e:
            logging.warning(f"Table extraction failed: {str(e)}")
        
        return tables
    
    def _extract_tables_tabula(self, pdf_path: str) -> List[Dict]:
        """Extract tables using tabula-py (if available)"""
        tables = []
        
        if not TABULA_AVAILABLE:
            logging.info("tabula-py not available, skipping tabula table extraction")
            return tables
            
        try:
            # Extract all tables from all pages
            dfs = tabula_read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            for i, df in enumerate(dfs):
                if not df.empty and df.shape[0] > 1:  # Has actual content
                    table_text = self._dataframe_to_text(df, f"Tabula Table {i+1}")
                    tables.append({
                        'source': 'tabula',
                        'table_id': f"tabula_{i+1}",
                        'content': table_text,
                        'dataframe': df
                    })
                    
        except Exception as e:
            # Only log if it's not the expected Java unavailable error
            if "java" not in str(e).lower():
                logging.warning(f"Tabula table extraction failed: {str(e)}")
            else:
                logging.info("Tabula table extraction skipped (Java not available, using pdfplumber)")
        
        return tables
    
    def _extract_tables_camelot(self, pdf_path: str) -> List[Dict]:
        """Extract tables using camelot (if available)"""
        tables = []
        
        if not CAMELOT_AVAILABLE:
            logging.info("Camelot not available, skipping camelot table extraction")
            return tables
            
        try:
            # Use lattice method for tables with borders
            camelot_tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(camelot_tables):
                if table.df.shape[0] > 1:  # Has actual content
                    table_text = self._dataframe_to_text(table.df, f"Camelot Table {i+1}")
                    tables.append({
                        'source': 'camelot_lattice',
                        'table_id': f"camelot_lattice_{i+1}",
                        'content': table_text,
                        'dataframe': table.df,
                        'accuracy': table.accuracy if hasattr(table, 'accuracy') else 0
                    })
            
            # Also try stream method for tables without borders
            try:
                stream_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
                for i, table in enumerate(stream_tables):
                    if table.df.shape[0] > 1:
                        table_text = self._dataframe_to_text(table.df, f"Camelot Stream Table {i+1}")
                        tables.append({
                            'source': 'camelot_stream',
                            'table_id': f"camelot_stream_{i+1}",
                            'content': table_text,
                            'dataframe': table.df,
                            'accuracy': table.accuracy if hasattr(table, 'accuracy') else 0
                        })
            except:
                pass  # Stream method might fail, continue
                
        except Exception as e:
            logging.warning(f"Camelot table extraction failed: {str(e)}")
        
        return tables
    
    def _extract_tables_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """Extract tables using pdfplumber"""
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for i, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Has headers and data
                            # Convert table to DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            table_text = self._dataframe_to_text(df, f"PDFPlumber Table (Page {page_num+1}, Table {i+1})")
                            
                            tables.append({
                                'source': 'pdfplumber',
                                'table_id': f"pdfplumber_p{page_num+1}_t{i+1}",
                                'content': table_text,
                                'dataframe': df,
                                'page': page_num + 1
                            })
                            
        except Exception as e:
            logging.warning(f"PDFPlumber table extraction failed: {str(e)}")
        
        return tables
    
    def _dataframe_to_text(self, df: pd.DataFrame, table_title: str) -> str:
        """Convert DataFrame to readable text format"""
        try:
            # Clean the dataframe
            df_clean = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df_clean.empty:
                return ""
            
            # Create a text representation
            text_parts = [f"\n{table_title}:"]
            
            # Add column headers
            headers = " | ".join(str(col) for col in df_clean.columns)
            text_parts.append(headers)
            text_parts.append("-" * len(headers))
            
            # Add rows
            for _, row in df_clean.iterrows():
                row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row)
                text_parts.append(row_text)
            
            text_parts.append("")  # Add blank line after table
            
            return "\n".join(text_parts)
            
        except Exception as e:
            logging.warning(f"Error converting DataFrame to text: {str(e)}")
            return f"\n{table_title}: [Table extraction error]\n"
    
    def _combine_text_and_tables(self, text_content: str, tables: List[Dict]) -> str:
        """Combine regular text with extracted table content"""
        if not tables:
            return text_content
        
        # Add all table content to the text
        combined_parts = [text_content]
        
        combined_parts.append("\n\n=== EXTRACTED TABLES ===\n")
        
        for table in tables:
            combined_parts.append(table['content'])
        
        return "\n".join(combined_parts)
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get total number of pages in PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except:
            return 0
    
    def _determine_processing_method(self, text: str) -> str:
        """Determine which method was used for text extraction"""
        if not text or len(text.strip()) < 100:
            return "failed"
        elif len(text) > 10000:
            return "pdfplumber"
        elif any(word in text.lower() for word in ['scanned', 'image', 'ocr']):
            return "ocr"
        else:
            return "pypdf2"
