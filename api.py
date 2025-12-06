"""
api.py
واجهة برمجية (API) محسّنة لبوت شركة الشمس تيليكوم باستخدام FastAPI
"""

import logging
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import html
import re

from rag_engine import get_answer, llm  # نفترض أن llm متاح للقراءة

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تهيئة التطبيق
app = FastAPI(
    title="Shams Telecom RAG Chatbot API",
    description="بوت ذكي للإجابة على أسئلة العملاء حول شركة الشمس تيليكوم",
    version="1.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, description="السؤال الذي يريد المستخدم إجابته")


class AnswerResponse(BaseModel):
    answer: str
    success: bool = True


def sanitize_question(question: str) -> str:
    """تنقية السؤال مع السماح بالرموز النصية العادية"""
    question = question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="السؤال فارغ.")
        
    if len(question) < 3:
        raise HTTPException(status_code=400, detail="السؤال قصير جدًّا. يرجى كتابة سؤال واضح.")
    
    # تنظيف HTML/JS (بدون رفض رموز نصية طبيعية)
    question = html.escape(question)
    # إزالة محاولات حقن بسيطة (بدون التأثير على الأسئلة الطبيعية)
    if re.search(r'(javascript:|<script|onload=|onerror=)', question, re.IGNORECASE):
        raise HTTPException(status_code=400, detail="السؤال يحتوي على محتوى غير آمن.")
    
    return question


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """تسجيل وقت البدء/الانتهاء لكل طلب"""
    logger.info(f"📥 وصول طلب: {request.method} {request.url.path}")
    start_time = asyncio.get_event_loop().time()
    response = await call_next(request)
    process_time = asyncio.get_event_loop().time() - start_time
    logger.info(f"📤 إرسال استجابة لـ {request.url.path} - الوقت: {process_time:.2f}s")
    return response


@app.post("/ask", response_model=AnswerResponse, summary="طرح سؤال والحصول على إجابة")
async def ask_question(request: QuestionRequest):
    try:
        clean_question = sanitize_question(request.question)
        
        # تنفيذ get_answer مع حد زمني (اختياري: يمكنك تفعيله إذا لزم)
        try:
            # يمكنك لاحقًا إضافة: asyncio.wait_for(get_answer(clean_question), timeout=10.0)
            answer = get_answer(clean_question)
        except Exception as e:
            logger.error(f"فشل في الحصول على إجابة: {e}")
            raise HTTPException(
                status_code=500, 
                detail="البوت لا يستطيع الرد حاليًا. يرجى المحاولة لاحقًا."
            )
        
        return AnswerResponse(answer=answer, success=True)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("خطأ غير متوقع في /ask")
        raise HTTPException(
            status_code=500, 
            detail="حدث خطأ غير متوقع. نعمل على إصلاحه."
        )


@app.get("/health")
async def health_check():
    """فحص صحة الخدمة مع معلومات دقيقة عن النموذج"""
    try:
        model_name = getattr(llm, 'model', 'unknown')
    except:
        model_name = "llama3"  # أو اقرأ من متغير عالمي
    
    return JSONResponse({
        "status": "online",
        "service": "Shams Telecom RAG Chatbot",
        "model": model_name,
        "retriever": "chroma_db",
        "version": "1.1.0",
        "ready": True
    })


@app.get("/")
async def root():
    return {
        "message": "مرحبًا بك في واجهة برمجية بوت شركة الشمس تيليكوم",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask"
    }