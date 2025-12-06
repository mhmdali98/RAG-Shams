"""
Pydantic models for API
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class QuestionRequest(BaseModel):
    """نموذج طلب السؤال"""
    question: str = Field(..., min_length=3, description="السؤال الذي يريد المستخدم إجابته")


class AnswerResponse(BaseModel):
    """نموذج الاستجابة"""
    answer: str = Field(..., description="إجابة البوت")
    success: bool = Field(default=True, description="حالة نجاح العملية")
    suggestions: Optional[List[str]] = None


class SuggestionsRequest(BaseModel):
    """نموذج طلب الاقتراحات"""
    question: Optional[str] = Field(None, description="السؤال الحالي (اختياري)")
    num_suggestions: Optional[int] = Field(4, ge=1, le=10, description="عدد الاقتراحات")


class SuggestionsResponse(BaseModel):
    """نموذج استجابة الاقتراحات"""
    suggestions: List[str]
    success: bool = True

