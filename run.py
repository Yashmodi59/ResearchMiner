#!/usr/bin/env python3
"""
ResearchMiner - AI Research Paper Analysis System
Launch script for local development
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import streamlit
        import openai
        import pdfplumber
        import PyPDF2
        import pandas
        import numpy
        import sklearn
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is configured"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠ Warning: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    print("✓ OpenAI API key is configured")
    return True

def main():
    """Main launch function"""
    print("🔬 Starting ResearchMiner - AI Research Paper Analysis System")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check API key
    check_api_key()
    
    # Launch Streamlit app
    print("\n🚀 Launching web interface...")
    print("📱 Access the application at: http://localhost:8501")
    print("💡 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down ResearchMiner...")

if __name__ == "__main__":
    main()