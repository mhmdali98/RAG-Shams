"""
api.py
واجهة برمجية (API) لبوت شركة الشمس تيليكوم باستخدام FastAPI
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from rag_engine import get_answer

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تهيئة التطبيق
app = FastAPI(
    title="Shams Telecom RAG Chatbot API",
    description="بوت ذكي للإجابة على أسئلة العملاء حول شركة الشمس تيليكوم",
    version="1.0.0"
)

# إعداد CORS للسماح بالوصول من أي مصدر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# نماذج البيانات
class QuestionRequest(BaseModel):
    """نموذج طلب السؤال"""
    question: str = Field(..., min_length=3, description="السؤال الذي يريد المستخدم إجابته")


class AnswerResponse(BaseModel):
    """نموذج الاستجابة"""
    answer: str = Field(..., description="إجابة البوت")
    success: bool = Field(default=True, description="حالة نجاح العملية")


def sanitize_question(question: str) -> str:
    """
    التحقق من صحة وتنقية السؤال
    
    Args:
        question: السؤال المدخل
        
    Returns:
        السؤال المنقى
        
    Raises:
        HTTPException: إذا كان السؤال غير صحيح
    """
    question = question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="السؤال فارغ.")
    
    if len(question) < 3:
        raise HTTPException(status_code=400, detail="السؤال قصير جدًّا. يرجى كتابة سؤال واضح.")
    
    # حماية من محاولات الحقن
    dangerous_chars = ["<", ">", "{", "}", "script", "alert", "javascript:"]
    if any(char in question.lower() for char in dangerous_chars):
        raise HTTPException(status_code=400, detail="السؤال يحتوي على محتوى غير آمن.")
    
    return question


@app.post("/ask", response_model=AnswerResponse, summary="طرح سؤال والحصول على إجابة")
async def ask_question(request: QuestionRequest):
    """
    نقطة النهاية الرئيسية لطرح الأسئلة والحصول على إجابات
    
    Args:
        request: طلب يحتوي على السؤال
        
    Returns:
        إجابة البوت مع حالة النجاح
    """
    try:
        logger.info(f"استلام سؤال: {request.question[:50]}...")
        
        # تنقية السؤال
        clean_question = sanitize_question(request.question)
        
        # الحصول على الإجابة من البوت
        answer = get_answer(clean_question)
        
        logger.info("تم إرجاع الإجابة بنجاح")
        return AnswerResponse(answer=answer, success=True)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"خطأ أثناء معالجة السؤال '{request.question}': {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى."
        )


@app.get("/health", summary="فحص حالة الخدمة")
async def health_check():
    """
    نقطة نهاية للتحقق من حالة الخدمة
    
    Returns:
        معلومات عن حالة الخدمة
    """
    return {
        "status": "online",
        "service": "Shams Telecom RAG Chatbot",
        "model": "mistral",
        "retriever": "chroma_db",
        "version": "1.0.0"
    }


@app.get("/", summary="الصفحة الرئيسية")
async def root():
    """الصفحة الرئيسية للـ API"""
    return {
        "message": "مرحبًا بك في واجهة برمجية بوت شركة الشمس تيليكوم",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask"
    }