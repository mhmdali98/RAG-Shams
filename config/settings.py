"""
إعدادات المشروع
"""

import os
from pathlib import Path

# مسارات المشروع
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DB_DIR = STORAGE_DIR / "chroma_db"

# إعدادات RAG
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.01"))
LLM_CONTEXT_SIZE = int(os.getenv("LLM_CONTEXT_SIZE", "4096"))

# إعدادات Retriever
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))
RETRIEVER_SCORE_THRESHOLD = float(os.getenv("RETRIEVER_SCORE_THRESHOLD", "0.2"))

# إعدادات API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Shams Telecom RAG Chatbot API"
API_VERSION = "2.0.0"

# إعدادات UI
UI_HOST = os.getenv("UI_HOST", "0.0.0.0")
UI_PORT = int(os.getenv("UI_PORT", "7860"))

# إعدادات البيانات
DATA_FILE = DATA_DIR / "shams-info.txt"

# إعدادات Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

class Settings:
    """فئة الإعدادات"""
    
    # المسارات
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    STORAGE_DIR = STORAGE_DIR
    CHROMA_DB_DIR = CHROMA_DB_DIR
    
    # RAG
    EMBEDDING_MODEL = EMBEDDING_MODEL
    LLM_MODEL = LLM_MODEL
    LLM_TEMPERATURE = LLM_TEMPERATURE
    LLM_CONTEXT_SIZE = LLM_CONTEXT_SIZE
    
    # Retriever
    RETRIEVER_K = RETRIEVER_K
    RETRIEVER_SCORE_THRESHOLD = RETRIEVER_SCORE_THRESHOLD
    
    # API
    API_HOST = API_HOST
    API_PORT = API_PORT
    API_TITLE = API_TITLE
    API_VERSION = API_VERSION
    
    # UI
    UI_HOST = UI_HOST
    UI_PORT = UI_PORT
    
    # البيانات
    DATA_FILE = DATA_FILE
    
    # Logging
    LOG_LEVEL = LOG_LEVEL

