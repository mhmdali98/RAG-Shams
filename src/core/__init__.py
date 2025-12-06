"""
Core components
"""

from .rag_engine import get_answer, vectorstore, retriever, llm
from .suggestions import get_suggestions, get_related_suggestions, detect_topic

__all__ = [
    'get_answer',
    'vectorstore',
    'retriever',
    'llm',
    'get_suggestions',
    'get_related_suggestions',
    'detect_topic'
]

