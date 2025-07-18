import re
import logging
from typing import List, Dict, Any

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}\"\'\`\~\@\#\$\%\^\&\*\+\=\|\\\/\<\>]', ' ', text)
    
    # Clean up line breaks
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    
    return text.strip()

def extract_paper_sections(text: str) -> Dict[str, str]:
    """
    Extract standard academic paper sections from text
    
    Args:
        text: Full paper text
        
    Returns:
        Dictionary with extracted sections
    """
    sections = {
        'title': '',
        'abstract': '',
        'introduction': '',
        'related_work': '',
        'methodology': '',
        'results': '',
        'discussion': '',
        'conclusion': '',
        'references': ''
    }
    
    # Define section patterns
    patterns = {
        'abstract': r'(?i)abstract\s*[:\-]?\s*(.*?)(?=introduction|1\.|methodology|related work|\n\n\w+:)',
        'introduction': r'(?i)(?:1\.?\s*)?introduction\s*[:\-]?\s*(.*?)(?=2\.|related work|methodology|background|\n\n\w+:)',
        'related_work': r'(?i)(?:2\.?\s*)?(?:related work|literature review|background)\s*[:\-]?\s*(.*?)(?=3\.|methodology|approach|\n\n\w+:)',
        'methodology': r'(?i)(?:3\.?\s*)?(?:methodology|methods?|approach|experimental setup)\s*[:\-]?\s*(.*?)(?=4\.|results|experiments?|evaluation|\n\n\w+:)',
        'results': r'(?i)(?:4\.?\s*)?(?:results?|experiments?|evaluation|findings)\s*[:\-]?\s*(.*?)(?=5\.|discussion|conclusion|\n\n\w+:)',
        'discussion': r'(?i)(?:5\.?\s*)?discussion\s*[:\-]?\s*(.*?)(?=6\.|conclusion|references|\n\n\w+:)',
        'conclusion': r'(?i)(?:6\.?\s*)?(?:conclusion|conclusions?|summary)\s*[:\-]?\s*(.*?)(?=references|acknowledgment|\n\n\w+:)',
        'references': r'(?i)references?\s*[:\-]?\s*(.*?)$'
    }
    
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            sections[section] = match.group(1).strip()[:2000]  # Limit length
    
    # Extract title (usually first significant text block)
    title_match = re.search(r'^(.{10,150})\n', text)
    if title_match:
        sections['title'] = title_match.group(1).strip()
    
    return sections

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def validate_pdf_file(file_content: bytes) -> bool:
    """
    Validate if file content is a valid PDF
    
    Args:
        file_content: File content as bytes
        
    Returns:
        True if valid PDF, False otherwise
    """
    # Check PDF header
    if file_content.startswith(b'%PDF-'):
        return True
    return False

def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using simple frequency analysis
    
    Args:
        text: Input text
        num_keywords: Number of keywords to extract
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Convert to lowercase and split
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Common stop words to exclude
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our',
        'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way',
        'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'that', 'with', 'have',
        'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time',
        'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take',
        'than', 'them', 'well', 'were', 'what', 'paper', 'study', 'research', 'analysis', 'method',
        'results', 'data', 'approach', 'using', 'based', 'model', 'proposed', 'experiment', 'evaluation'
    }
    
    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count frequency
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_words[:num_keywords]]

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time for text
    
    Args:
        text: Input text
        words_per_minute: Average reading speed
        
    Returns:
        Estimated reading time in minutes
    """
    if not text:
        return 0
    
    word_count = len(text.split())
    reading_time = max(1, word_count // words_per_minute)
    
    return reading_time

def chunk_text_by_sentences(text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Chunk text by sentences while respecting size limits
    
    Args:
        text: Input text
        max_chunk_size: Maximum characters per chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed the limit
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Initialize logging when module is imported
setup_logging()
