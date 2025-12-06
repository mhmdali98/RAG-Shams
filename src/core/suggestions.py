"""
suggestions.py
نظام توليد اقتراحات ذكية للأسئلة
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# اقتراحات ثابتة بناءً على المواضيع
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
        "هل تقدمون إنترنت عبر الألياف الضوئية？",
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
    
    keywords = {
        "باقات": ["باقة", "باقات", "اشتراك", "خطة", "star", "sun", "neptune", "فايبر", "وايرلس"],
        "أسعار": ["سعر", "أسعار", "كم", "دينار", "تكلفة", "ثمن", "سعر"],
        "تغطية": ["تغطية", "منطقة", "فرع", "مكان", "أين", "بغداد", "ديالى", "بابل"],
        "دعم": ["دعم", "مساعدة", "خدمة", "اتصال", "24", "فني"],
        "خدمات": ["خدمة", "خدمات", "إنترنت", "ftth", "wifi", "بلو سيركل"],
        "شركة": ["شركة", "من نحن", "عن", "قيم", "شركاء", "خبرة"],
        "تواصل": ["تواصل", "اتصال", "هاتف", "بريد", "واتساب", "موقع", "عنوان"]
    }
    
    scores = {}
    for topic, topic_keywords in keywords.items():
        score = sum(1 for keyword in topic_keywords if keyword in q_lower)
        if score > 0:
            scores[topic] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    return "عام"


def get_suggestions(question: str = "", context: str = "", num_suggestions: int = 4) -> List[str]:
    """توليد اقتراحات ذكية"""
    suggestions = []
    
    if question:
        topic = detect_topic(question)
        topic_suggestions = STATIC_SUGGESTIONS.get(topic, [])
        
        for sug in topic_suggestions:
            if sug not in suggestions and sug != question:
                suggestions.append(sug)
                if len(suggestions) >= num_suggestions:
                    break
        
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
        suggestions = GENERAL_SUGGESTIONS[:num_suggestions]
    
    unique_suggestions = []
    seen = set()
    for sug in suggestions:
        if sug not in seen:
            seen.add(sug)
            unique_suggestions.append(sug)
    
    return unique_suggestions[:num_suggestions]


def get_related_suggestions(question: str, vectorstore=None, num_suggestions: int = 3) -> List[str]:
    """توليد اقتراحات بناءً على استرجاع مشابه"""
    if not vectorstore:
        return get_suggestions(question, num_suggestions=num_suggestions)
    
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        docs = retriever.invoke(question)
        
        keywords = set()
        for doc in docs:
            content = doc.page_content.lower()
            important_words = ["باقة", "سعر", "دينار", "فايبر", "وايرلس", "تغطية", "دعم", "خدمة"]
            for word in important_words:
                if word in content:
                    keywords.add(word)
        
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
        
        if len(suggestions) < num_suggestions:
            suggestions.extend(GENERAL_SUGGESTIONS[:num_suggestions - len(suggestions)])
        
        return suggestions[:num_suggestions]
        
    except Exception as e:
        logger.error(f"خطأ في توليد الاقتراحات: {e}")
        return get_suggestions(question, num_suggestions=num_suggestions)

