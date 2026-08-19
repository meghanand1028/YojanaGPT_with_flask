import sys
import os

# Add parent directory to path so imports work cleanly on Vercel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Export WSGI application for Vercel Serverless Functions
app = app
