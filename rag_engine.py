"""
rag_engine.py
محرك RAG محسّن للإجابة الدقيقة والاحترافية على أسئلة العملاء
حول شركة الشمس تيليكوم.
"""

import logging
import re
from typing import List, Optional, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === تهيئة النظام ===
try:
    logger.info("جاري تحميل نموذج التضمين...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    logger.info("جاري تحميل قاعدة البيانات المتجهة...")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    
    # Retriever ذكي: يجمع بين k عالي وعتبة تشابه محسّنة
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.2}  # زيادة k وتحسين العتبة
    )
    
    # Retriever احتياطي للبحث الواسع
    fallback_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    logger.info("جاري تهيئة النموذج اللغوي...")
    llm = ChatOllama(model="llama3", temperature=0.01, num_ctx=4096)  # أقل حرارة = أكثر دقة
    
    logger.info("✅ تم تهيئة النظام بنجاح")
    
except Exception as e:
    logger.error(f"❌ خطأ في تهيئة النظام: {str(e)}")
    raise

# تصدير vectorstore للاستخدام في modules أخرى
__all__ = ['get_answer', 'vectorstore', 'retriever', 'llm']

# قالب تعليمات محسّن ومحترف
prompt = ChatPromptTemplate.from_messages([
    ("system", """أنت مساعد ذكي احترافي لشركة "الشمس تيليكوم". قواعدك الصارمة:

**القواعد الأساسية:**
1. **الإجابة باللغة العربية فقط** - ممنوع استخدام أي لغة أخرى.
2. **استخلص المعلومات مباشرة من السياق** - ممنوع الاختلاق أو الإضافة من خارج السياق.
3. **إذا لم تجد المعلومة في السياق**، قل بوضوح: "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية."

**عند السؤال عن الباقات أو الأسعار:**
- اذكر **جميع** الباقات المتوفرة من السياق
- لكل باقة: الاسم، السعر (بالدينار العراقي)، النوع (فايبر/وايرلس/منصة)، السرعة (إن وجدت)، وصف موجز
- نظمها بشكل واضح باستخدام نقاط أو جداول

**عند السؤال عن الخدمات:**
- اذكر جميع الخدمات المتوفرة من السياق
- اذكر التفاصيل المهمة (مثل: 70,000 خط كابل ضوئي)

**عند السؤال عن التواصل:**
- اذكر: الهاتف (6449)، البريد (info@shams-tele.com)، المواقع، وسائل التواصل الاجتماعي

**أسلوب الإجابة:**
- احترافي، ودود، وواضح
- استخدم عناوين فرعية أو نقاط عند الحاجة
- لا تكرر نفس المعلومة
- كن موجزًا لكن شاملًا

**ممنوع تمامًا:**
- الاختلاق أو الإضافة من خارج السياق
- استخدام لغات غير العربية
- إعطاء معلومات غير موجودة في السياق
- التكرار المفرط

**تأكد من:**
- الإجابة بالعربية فقط
- جميع المعلومات من السياق فقط
- الدقة في الأرقام والأسماء
"""),
    ("human", """**السياق المتاح (استخدمه فقط كمصدر للمعلومات):**
{context}

**سؤال العميل:**
{input}

**أجب باللغة العربية فقط، واستخدم المعلومات من السياق أعلاه حصريًا:**""")
])


def filter_and_deduplicate_docs(docs: List[Document]) -> str:
    """
    تصفية المستندات غير المفيدة + إزالة التكرار + الحد من الطول
    """
    if not docs:
        return "لا توجد معلومات كافية."
    
    # استبعاد أقسام الأخبار الطويلة (غير مباشرة للاستعلام)
    filtered = []
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
            
        # استبعاد إذا كان يحتوي على كلمات مثل "شاركنا في فعالية" أو "ورشة عمل"
        if any(term in content for term in ["شاركنا في فعالية", "ورشة عمل", "رعاة", "حدث", "مؤتمر"]):
            if len(content) > 300:
                # نبقي فقط جملة موجزة إن وُجدت
                first_line = content.split("\n")[0]
                if any(keyword in first_line for keyword in ["شمس", "باقة", "سعر", "دينار", "FTTH", "WiFi"]):
                    filtered.append(first_line)
                continue
        filtered.append(content)
    
    # إزالة التكرار تقريبيًا (محسّن)
    unique_contents = []
    seen = set()
    for text in filtered:
        # استخدام hash أفضل للتكرار
        text_lower = text.lower().strip()
        # إنشاء مفتاح فريد من أول 100 حرف
        key = text_lower[:100] if len(text_lower) > 100 else text_lower
        if key not in seen:
            seen.add(key)
            unique_contents.append(text)
    
    # دمج النص مع حد أقصى ~2000 حرف (زيادة للدقة)
    context = "\n---\n".join(unique_contents)
    return context[:2000]


# سلسلة RAG محسّنة
rag_chain = (
    {"context": retriever | filter_and_deduplicate_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def is_arabic_text(text: str) -> bool:
    """فحص إذا كان النص عربيًا بشكل أساسي"""
    if not text or not text.strip():
        return False
    
    # حساب نسبة الأحرف العربية
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF' or char in '،؛؟')
    total_chars = len([c for c in text if c.isalpha() or c in '،؛؟'])
    
    if total_chars == 0:
        return False
    
    arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
    return arabic_ratio >= 0.3  # على الأقل 30% عربي


def validate_answer(answer: str, context: str) -> Tuple[bool, str]:
    """التحقق من صحة الإجابة"""
    if not answer or len(answer.strip()) < 10:
        return False, "الإجابة قصيرة جدًا"
    
    # فحص اللغة العربية
    if not is_arabic_text(answer):
        return False, "الإجابة ليست بالعربية"
    
    # فحص وجود معلومات مفيدة
    if any(phrase in answer.lower() for phrase in ["لا أعرف", "لا أملك", "لا يوجد", "غير متوفر"]):
        if "عذرًا" not in answer and "قاعدة بياناتنا" not in answer:
            return False, "الإجابة غير واضحة"
    
    # فحص التكرار المفرط
    words = answer.split()
    if len(set(words)) < len(words) * 0.3:  # أكثر من 70% تكرار
        return False, "الإجابة تحتوي على تكرار مفرط"
    
    return True, "صحيحة"


def expand_query(question: str) -> str:
    """توسيع ذكي للسؤال بناءً على أنماط شمس تيليكوم"""
    q = question.strip().lower()
    original = question.strip()
    
    # أنماط البحث الشائعة (محسّنة)
    if any(w in q for w in ["السعر", "سعر", "كم", "دينار", "تكلفة", "ثمن"]):
        return f"{original} باقات الإنترنت الأسعار FTTH WiFi دينار عراقي فايبر وايرلس"
    
    if any(w in q for w in ["باقة", "اشتراك", "الباقات", "باقات", "خطة"]):
        return f"{original} باقات الإنترنت FTTH WiFi المنصة فايبر Star Sun Neptune Galaxy Star"
    
    if any(w in q for w in ["تغطية", "منطقة", "أين", "فرع", "مكان"]):
        return f"{original} تغطية بغداد ديالى بابل المحافظات فرع موقع"
    
    if any(w in q for w in ["دعم", "مساعده", "24", "خدمة", "اتصال"]):
        return f"{original} دعم فني 24/7 خدمة العملاء تواصل"
    
    if any(w in q for w in ["شركة", "من نحن", "عن", "معلومات"]):
        return f"{original} شركة شمس تيليكوم معلومات عن الشركة"
    
    return original


def get_answer(question: str, max_retries: int = 2) -> str:
    """
    الحصول على إجابة دقيقة وسريعة مع نظام إعادة المحاولة والتحقق
    """
    if not question or not question.strip():
        return "مرحباً! 🌞 كيف يمكنني مساعدتك اليوم؟ اسألني عن باقات الإنترنت، الأسعار، التغطية، أو أي معلومات عن شمس تيليكوم."

    clean_q = question.strip()
    logger.info(f"معالجة السؤال: '{clean_q}'")

    # فحص اللغة العربية للسؤال
    if not is_arabic_text(clean_q):
        logger.warning(f"السؤال قد لا يكون بالعربية: '{clean_q}'")

    # توسيع الاستعلام
    expanded = expand_query(clean_q)
    if expanded != clean_q:
        logger.debug(f"السؤال الموسّع: {expanded}")

    context_used = ""
    
    try:
        # استرجاع أولي
        docs = retriever.invoke(expanded)
        logger.info(f"تم استرجاع {len(docs)} مستند(ات)")

        # إذا فشل الاسترجاع، نحاول بخطوات احتياطية
        if not docs:
            fallback_queries = [
                clean_q,
                " ".join([w for w in clean_q.split() if len(w) > 2]),
                "باقات أسعار دعم تواصل"
            ]
            for fq in fallback_queries:
                docs = retriever.invoke(fq)
                if docs:
                    logger.info(f"تم الاسترجاع باستخدام الاستعلام الاحتياطي: {fq}")
                    break
            
            # إذا لم نجد شيئًا، نستخدم retriever احتياطي
            if not docs:
                docs = fallback_retriever.invoke(clean_q)
                logger.info(f"تم الاسترجاع باستخدام retriever احتياطي: {len(docs)} مستند(ات)")

        # تحضير السياق
        context_used = filter_and_deduplicate_docs(docs)
        
        # تنفيذ السلسلة مع إعادة المحاولة
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = rag_chain.invoke(clean_q)
                response = response.strip()
                
                # التحقق من صحة الإجابة
                is_valid, validation_msg = validate_answer(response, context_used)
                if is_valid:
                    logger.info(f"تم التحقق من صحة الإجابة (المحاولة {attempt + 1})")
                    break
                else:
                    logger.warning(f"الإجابة غير صحيحة: {validation_msg} (المحاولة {attempt + 1})")
                    if attempt < max_retries:
                        # إعادة المحاولة مع prompt محسّن
                        continue
                    else:
                        # إذا فشلت جميع المحاولات، نعيد صياغة الإجابة
                        response = "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية.\n\nهل تريد معرفة معلومات عن باقاتنا أو خدماتنا؟ 😊"
            except Exception as e:
                logger.error(f"خطأ في المحاولة {attempt + 1}: {e}")
                if attempt < max_retries:
                    continue
                else:
                    raise
        
        # تنظيف نهائي للإجابة
        if not response:
            response = "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية.\n\nهل تريد معرفة معلومات عن باقاتنا أو خدماتنا؟ 😊"
        
        # إزالة أي نص إنجليزي غير ضروري (مثل "Answer:" أو "Response:")
        response = re.sub(r'^(Answer|Response|Reply):\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        # التأكد من أن الإجابة بالعربية
        if not is_arabic_text(response):
            logger.warning("الإجابة قد تحتوي على نص غير عربي، سيتم إعادة المحاولة...")
            # محاولة أخيرة
            try:
                response = rag_chain.invoke(f"{clean_q}\n\nتأكد من الإجابة باللغة العربية فقط.")
                response = response.strip()
            except:
                pass
        
        return response

    except Exception as e:
        logger.error(f"خطأ في معالجة السؤال: {e}", exc_info=True)
        return "عذرًا، حدث خطأ تقني. يرجى المحاولة لاحقًا أو الاتصال بنا على 6449."