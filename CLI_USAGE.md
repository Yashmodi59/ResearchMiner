# CLI Usage Guide

## Overview

The CLI interface provides command-line access to the AI Research Paper Analysis system without needing the web interface.

## Installation & Setup

Make sure you have the OpenAI API key set:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Basic Usage

### 1. Process a Single Document
```bash
python cli.py process paper.pdf
```

### 2. Process All PDFs in a Directory
```bash
python cli.py process-dir /path/to/papers/
```

### 3. Query Documents (after processing)
```bash
python cli.py query "What methodology was used in the study?"
```

### 4. Interactive Mode
```bash
python cli.py interactive
```

### 5. List Loaded Documents
```bash
python cli.py list
```

## Advanced Usage

### Process and Query in One Command
```bash
python cli.py process paper.pdf --query "What are the main findings?"
```

### Process Directory and Query
```bash
python cli.py process-dir ./papers/ --query "Compare the methodologies used"
```

### Specify Query Type
```bash
python cli.py query "Compare results across papers" --query-type "Cross-Paper Comparison"
```

### Get More Relevant Chunks
```bash
python cli.py query "methodology details" --top-k 10
```

### Show Debug Information
```bash
python cli.py query "results" --show-debug
```

### Save Session Data
```bash
python cli.py process-dir ./papers/ --save-session session.json
```

## Interactive Mode Commands

Once in interactive mode, you can use:
- `list` - Show all loaded documents
- `help` - Show available commands  
- `quit`, `exit`, or `q` - Exit interactive mode
- Any text question to query the documents

## Command Reference

### Commands
- `process <file>` - Process a single PDF file
- `process-dir <directory>` - Process all PDFs in directory
- `query <question>` - Query the loaded documents
- `interactive` - Start interactive query mode
- `list` - List all loaded documents

### Options
- `--query, -q <text>` - Run query after processing
- `--query-type <type>` - Specify query type:
  - `auto` (default)
  - `Direct Content Lookup`
  - `Cross-Paper Comparison`
  - `Methodology Summary`
  - `Results Extraction`
  - `Conclusion Summary`
- `--top-k <number>` - Number of chunks to retrieve (default: 5)
- `--show-sources` - Show source information (default: true)
- `--show-debug` - Show search debug information
- `--save-session <file>` - Save session data to JSON file
- `--verbose, -v` - Enable verbose logging

## Example Workflows

### Research Paper Review
```bash
# Process all papers in a folder
python cli.py process-dir ./research_papers/

# Start interactive analysis
python cli.py interactive

# In interactive mode:
Query> What are the main methodologies used across all papers?
Query> Compare the results between papers
Query> What limitations are mentioned?
Query> quit
```

### Single Document Analysis
```bash
# Process and immediately query
python cli.py process paper.pdf \
  --query "What is the main contribution of this work?" \
  --show-debug

# Follow up with more questions
python cli.py query "What are the experimental results?"
python cli.py query "What future work is suggested?"
```

### Batch Processing with Session Save
```bash
# Process directory and save session
python cli.py process-dir ./papers/ \
  --query "Summarize the key findings from all papers" \
  --save-session analysis_session.json \
  --verbose
```

## Output Format

### Processing Output
```
Processing paper.pdf...
✓ Successfully processed paper.pdf
  - Pages: 12
  - Text chunks: 45
  - Tables found: 3
  - Characters: 28,542
  - Method: pdfplumber
```

### Query Response Output
```
================================================================================
ANSWER:
================================================================================
The study used a mixed-methods approach combining quantitative analysis...

--------------------------------------------------------------------------------
SOURCES:
--------------------------------------------------------------------------------

[1] paper.pdf (Page 3)
    Relevance Score: 0.850
    Search Methods: semantic: 0.78, keyword: 0.65, fuzzy: 0.42
    Preview: The methodology section describes the experimental design...

[2] paper.pdf (Page 4)
    Relevance Score: 0.720
    Preview: Additional methodological details include the data collection...

================================================================================
```

## Error Handling

The CLI provides clear error messages for common issues:
- File not found
- Directory doesn't exist
- No documents loaded
- API key not set
- Processing failures

## Performance Notes

- Processing large PDFs may take time
- Table extraction adds processing overhead
- Interactive mode keeps documents in memory
- Use `--top-k` to control response time vs accuracy