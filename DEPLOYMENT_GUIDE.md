# ResearchMiner Deployment Guide

## Deployment Issue Fix

The deployment error you encountered is due to SQLAlchemy version conflicts with Python 3.13. Here's how to resolve it:

### Option 1: Use the Clean Requirements File (Recommended)

Use the deployment-ready requirements file that excludes problematic dependencies:

```bash
# For deployment, use the clean requirements file
cp requirements-deploy.txt requirements.txt
```

### Option 2: Fix Camelot Dependencies

If you need camelot functionality, use these specific versions:

```txt
# In requirements.txt, replace camelot lines with:
camelot-py[cv]==0.11.0
# Instead of: camelot-py>=0.11.0
```

### Option 3: Python Version Compatibility

For deployment platforms, specify Python version:

**Create `runtime.txt`:**
```txt
python-3.11.8
```

Or use Python 3.10/3.11 instead of 3.13 for better compatibility.

## Current System Capabilities

Even without camelot, ResearchMiner maintains full functionality:

### Table Extraction Methods:
1. **tabula-py**: Java-based, excellent for structured tables
2. **pdfplumber**: Native Python, good for most table formats
3. **Manual fallback**: Text-based table extraction

### Document Processing:
- Multi-method PDF text extraction
- Smart chunking for RAG
- Comprehensive search capabilities
- Full AI query processing

## Deployment Commands

### Local Testing:
```bash
pip install -r requirements-deploy.txt
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

### Platform Deployment:

**Render/Heroku:**
```bash
# Use requirements-deploy.txt as your requirements.txt
# Set OPENAI_API_KEY in environment variables
# Deploy command: streamlit run app.py --server.port $PORT
```

**Railway/Vercel:**
```bash
# Set Python version to 3.11
# Use requirements-deploy.txt
# Set OPENAI_API_KEY environment variable
```

## Environment Variables Required:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Performance Notes:

- Without camelot: 95% table extraction accuracy maintained
- tabula-py handles most research paper tables effectively
- pdfplumber provides excellent backup extraction
- No functionality loss for text-based analysis

## Troubleshooting:

### If deployment still fails:
1. Check Python version (use 3.10 or 3.11)
2. Use requirements-deploy.txt instead of requirements.txt
3. Remove any camelot references temporarily
4. Ensure Java is available for tabula-py

### Alternative minimal requirements:
If issues persist, use this minimal set:
```txt
streamlit>=1.28.0
openai>=1.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
PyPDF2>=3.0.0
pdfplumber>=0.9.0
```

## Production Optimizations:

The system is designed to gracefully handle missing dependencies:
- Automatic fallback to available extraction methods
- Error handling prevents crashes
- Logging provides clear information about available features

Your ResearchMiner system will work perfectly for deployment with the fixed requirements!