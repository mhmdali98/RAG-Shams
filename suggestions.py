"""
suggestions.py
نظام توليد اقتراحات ذكية للأسئلة بناءً على السياق والموضوع
"""

import logging
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# اقتراحات ثابتة بناءً على المواضيع الشائعة
STATIC_SUGGESTIONS = {
    "باقات": [
        "ما هي جميع باقات الإنترنت المتوفرة؟",
        "ما سعر باقة فايبر 75؟",
        "ما الفرق بين باقات الفايبر والوايرلس؟",
        "ما هي باقات المنصة الترفيهية؟"
    ],
    "أسعار": [
        "ما هي أسعار جميع الباقات؟",
        "كم سعر باقة Sun؟",
        "ما سعر باقة المنصة لثلاثة أشهر؟",
        "ما هي أرخص باقة متوفرة؟"
    ],
    "تغطية": [
        "ما هي مناطق التغطية؟",
        "هل لديكم فرع في ديالى؟",
        "أين تقع فروعكم؟",
        "ما هي المحافظات المغطاة؟"
    ],
    "دعم": [
        "هل الدعم الفني متاح 24 ساعة؟",
        "كيف أتواصل مع الدعم الفني؟",
        "ما هو رقم خدمة العملاء؟"
    ],
    "خدمات": [
        "ما هي الخدمات التي تقدمونها؟",
        "هل تقدمون إنترنت عبر الألياف الضوئية؟",
        "ما هي خدمة بلو سيركل؟"
    ],
    "شركة": [
        "من نحن؟",
        "ما هي قيم الشركة؟",
        "من هم شركاؤكم؟",
        "كم سنة من الخبرة لديكم؟"
    ],
    "تواصل": [
        "كيف أتواصل معكم؟",
        "ما هو رقم الهاتف؟",
        "ما هو البريد الإلكتروني؟",
        "أين موقعكم؟"
    ]
}

# اقتراحات عامة
GENERAL_SUGGESTIONS = [
    "ما هي جميع باقات الإنترنت المتوفرة؟",
    "ما هي أسعار الباقات؟",
    "ما هي مناطق التغطية؟",
    "كيف أتواصل معكم؟",
    "هل الدعم الفني متاح 24 ساعة؟",
    "ما هي الخدمات التي تقدمونها؟"
]


def detect_topic(question: str) -> str:
    """اكتشاف موضوع السؤال"""
    q_lower = question.lower()
    
    # كلمات مفتاحية لكل موضوع
    keywords = {
        "باقات": ["باقة", "باقات", "اشتراك", "خطة", "star", "sun", "neptune", "فايبر", "وايرلس"],
        "أسعار": ["سعر", "أسعار", "كم", "دينار", "تكلفة", "ثمن", "سعر"],
        "تغطية": ["تغطية", "منطقة", "فرع", "مكان", "أين", "بغداد", "ديالى", "بابل"],
        "دعم": ["دعم", "مساعدة", "خدمة", "اتصال", "24", "فني"],
        "خدمات": ["خدمة", "خدمات", "إنترنت", "ftth", "wifi", "بلو سيركل"],
        "شركة": ["شركة", "من نحن", "عن", "قيم", "شركاء", "خبرة"],
        "تواصل": ["تواصل", "اتصال", "هاتف", "بريد", "واتساب", "موقع", "عنوان"]
    }
    
    # حساب النقاط لكل موضوع
    scores = {}
    for topic, topic_keywords in keywords.items():
        score = sum(1 for keyword in topic_keywords if keyword in q_lower)
        if score > 0:
            scores[topic] = score
    
    # إرجاع الموضوع الأعلى نقاطًا
    if scores:
        return max(scores, key=scores.get)
    
    return "عام"


def get_suggestions(question: str = "", context: str = "", num_suggestions: int = 4) -> List[str]:
    """
    توليد اقتراحات ذكية بناءً على السؤال والسياق
    
    Args:
        question: السؤال الحالي (اختياري)
        context: السياق المسترجع (اختياري)
        num_suggestions: عدد الاقتراحات المطلوبة
    
    Returns:
        قائمة بالاقتراحات
    """
    suggestions = []
    
    # إذا كان هناك سؤال، نكتشف الموضوع ونعطي اقتراحات متعلقة
    if question:
        topic = detect_topic(question)
        topic_suggestions = STATIC_SUGGESTIONS.get(topic, [])
        
        # إضافة اقتراحات الموضوع
        for sug in topic_suggestions:
            if sug not in suggestions and sug != question:
                suggestions.append(sug)
                if len(suggestions) >= num_suggestions:
                    break
        
        # إذا لم نملأ العدد المطلوب، نضيف من مواضيع أخرى
        if len(suggestions) < num_suggestions:
            for other_topic, other_suggestions in STATIC_SUGGESTIONS.items():
                if other_topic != topic:
                    for sug in other_suggestions:
                        if sug not in suggestions and sug != question:
                            suggestions.append(sug)
                            if len(suggestions) >= num_suggestions:
                                break
                    if len(suggestions) >= num_suggestions:
                        break
    else:
        # إذا لم يكن هناك سؤال، نعطي اقتراحات عامة
        suggestions = GENERAL_SUGGESTIONS[:num_suggestions]
    
    # التأكد من عدم وجود تكرار
    unique_suggestions = []
    seen = set()
    for sug in suggestions:
        if sug not in seen:
            seen.add(sug)
            unique_suggestions.append(sug)
    
    return unique_suggestions[:num_suggestions]


def get_related_suggestions(question: str, vectorstore: Chroma = None, num_suggestions: int = 3) -> List[str]:
    """
    توليد اقتراحات بناءً على استرجاع مشابه من قاعدة البيانات
    
    Args:
        question: السؤال الحالي
        vectorstore: قاعدة البيانات المتجهة (اختياري)
        num_suggestions: عدد الاقتراحات
    
    Returns:
        قائمة بالاقتراحات
    """
    if not vectorstore:
        return get_suggestions(question, num_suggestions=num_suggestions)
    
    try:
        # البحث عن مواضيع مشابهة
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        docs = retriever.invoke(question)
        
        # استخراج كلمات مفتاحية من المستندات المسترجعة
        keywords = set()
        for doc in docs:
            content = doc.page_content.lower()
            # كلمات مفتاحية محتملة
            important_words = ["باقة", "سعر", "دينار", "فايبر", "وايرلس", "تغطية", "دعم", "خدمة"]
            for word in important_words:
                if word in content:
                    keywords.add(word)
        
        # توليد اقتراحات بناءً على الكلمات المفتاحية
        suggestions = []
        if "باقة" in keywords or "سعر" in keywords:
            suggestions.extend([
                "ما هي جميع باقات الإنترنت المتوفرة؟",
                "ما هي أسعار الباقات؟"
            ])
        if "تغطية" in keywords:
            suggestions.append("ما هي مناطق التغطية؟")
        if "دعم" in keywords:
            suggestions.append("هل الدعم الفني متاح 24 ساعة؟")
        
        # إذا لم نجد اقتراحات كافية، نضيف عامة
        if len(suggestions) < num_suggestions:
            suggestions.extend(GENERAL_SUGGESTIONS[:num_suggestions - len(suggestions)])
        
        return suggestions[:num_suggestions]
        
    except Exception as e:
        logger.error(f"خطأ في توليد الاقتراحات: {e}")
        return get_suggestions(question, num_suggestions=num_suggestions)

