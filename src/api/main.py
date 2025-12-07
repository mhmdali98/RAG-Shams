
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import Settings
from src.core import get_answer, vectorstore
from src.core.suggestions import get_suggestions, get_related_suggestions
from .models import QuestionRequest, AnswerResponse, SuggestionsRequest, SuggestionsResponse

# إعداد السجلات
logging.basicConfig(level=getattr(logging, Settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# تهيئة التطبيق
app = FastAPI(
    title=Settings.API_TITLE,
    description="بوت ذكي للإجابة على أسئلة العملاء حول شركة الشمس تيليكوم",
    version=Settings.API_VERSION
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def sanitize_question(question: str) -> str:
    """التحقق من صحة وتنقية السؤال"""
    question = question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="السؤال فارغ.")
    
    if len(question) < 3:
        raise HTTPException(status_code=400, detail="السؤال قصير جدًّا.")
    
    dangerous_chars = ["<", ">", "{", "}", "script", "alert", "javascript:"]
    if any(char in question.lower() for char in dangerous_chars):
        raise HTTPException(status_code=400, detail="السؤال يحتوي على محتوى غير آمن.")
    
    return question


@app.post("/ask", response_model=AnswerResponse, summary="طرح سؤال والحصول على إجابة")
async def ask_question(request: QuestionRequest):
    """طرح سؤال والحصول على إجابة مع اقتراحات"""
    try:
        clean_question = sanitize_question(request.question)
        answer = get_answer(clean_question)
        
        try:
            suggestions = get_related_suggestions(clean_question, vectorstore, num_suggestions=4)
        except:
            suggestions = get_suggestions(clean_question, num_suggestions=4)
        
        return AnswerResponse(answer=answer, success=True, suggestions=suggestions)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("خطأ في /ask")
        raise HTTPException(status_code=500, detail="حدث خطأ غير متوقع.")


@app.post("/suggestions", response_model=SuggestionsResponse, summary="الحصول على اقتراحات")
async def get_suggestions_endpoint(request: SuggestionsRequest):
    """الحصول على اقتراحات ذكية"""
    try:
        if request.question:
            clean_question = sanitize_question(request.question)
            suggestions = get_related_suggestions(
                clean_question,
                vectorstore,
                num_suggestions=request.num_suggestions
            )
        else:
            suggestions = get_suggestions("", num_suggestions=request.num_suggestions)
        
        return SuggestionsResponse(suggestions=suggestions, success=True)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("خطأ في /suggestions")
        raise HTTPException(status_code=500, detail="حدث خطأ في توليد الاقتراحات.")


@app.get("/health")
async def health_check():
    """فحص حالة الخدمة"""
    return JSONResponse({
        "status": "online",
        "service": "Shams Telecom RAG Chatbot",
        "model": Settings.LLM_MODEL,
        "retriever": "chroma_db",
        "version": Settings.API_VERSION,
        "ready": True
    })


@app.get("/")
async def root():
    """الصفحة الرئيسية"""
    return {
        "message": "مرحبًا بك في واجهة برمجية بوت شركة الشمس تيليكوم",
        "version": Settings.API_VERSION,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "ask": "/ask",
            "suggestions": "/suggestions"
        }
    }

