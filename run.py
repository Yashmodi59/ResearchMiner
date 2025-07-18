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
    
    # Get port from environment variable (for deployment) or use default
    port = os.getenv('PORT', '8501')
    
    # Determine if running in production (Render, Heroku, etc.)
    is_production = bool(os.getenv('PORT') or os.getenv('RENDER') or os.getenv('DYNO'))
    
    if is_production:
        # Production deployment
        address = "0.0.0.0"
        print(f"\n🚀 Launching production server on port {port}...")
    else:
        # Local development
        address = "localhost"
        print(f"\n🚀 Launching web interface...")
        print(f"📱 Access the application at: http://localhost:{port}")
        print("💡 Press Ctrl+C to stop the server")
    
    print("=" * 60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", port,
            "--server.address", address,
            "--server.headless", "true" if is_production else "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down ResearchMiner...")

if __name__ == "__main__":
    main()