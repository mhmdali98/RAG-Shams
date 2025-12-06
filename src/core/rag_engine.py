"""
rag_engine.py
محرك RAG (Retrieval-Augmented Generation) للإجابة على أسئلة العملاء
حول شركة الشمس تيليكوم بناءً على البيانات المحفوظة في قاعدة البيانات المتجهة.
"""

import logging
import sys
from pathlib import Path

# إضافة مسار المشروع إلى Python path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import Settings

# إعداد السجلات
logging.basicConfig(level=getattr(logging, Settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

# === تهيئة النظام ===
try:
    logger.info("جاري تحميل نموذج التضمين...")
    embeddings = HuggingFaceEmbeddings(
        model_name=Settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    logger.info("جاري تحميل قاعدة البيانات المتجهة...")
    vectorstore = Chroma(
        persist_directory=str(Settings.CHROMA_DB_DIR),
        embedding_function=embeddings
    )
    
    # Retriever محسّن
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": Settings.RETRIEVER_K,
            "score_threshold": Settings.RETRIEVER_SCORE_THRESHOLD
        }
    )
    
    # Retriever احتياطي
    fallback_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    logger.info("جاري تهيئة النموذج اللغوي...")
    llm = ChatOllama(
        model=Settings.LLM_MODEL,
        temperature=Settings.LLM_TEMPERATURE,
        num_ctx=Settings.LLM_CONTEXT_SIZE
    )
    
    logger.info("✅ تم تهيئة النظام بنجاح")
    
except Exception as e:
    logger.error(f"❌ خطأ في تهيئة النظام: {str(e)}")
    raise

# قالب التعليمات المحسّن
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


def filter_and_deduplicate_docs(docs) -> str:
    """تصفية المستندات وإزالة التكرار"""
    if not docs:
        return "لا توجد معلومات كافية."
    
    filtered = []
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
        
        # تصفية المحتوى غير المباشر
        if any(term in content for term in ["شاركنا في فعالية", "ورشة عمل", "رعاة", "حدث", "مؤتمر"]):
            if len(content) > 300:
                first_line = content.split("\n")[0]
                if any(keyword in first_line for keyword in ["شمس", "باقة", "سعر", "دينار", "FTTH", "WiFi"]):
                    filtered.append(first_line)
                continue
        filtered.append(content)
    
    # إزالة التكرار
    unique_contents = []
    seen = set()
    for text in filtered:
        text_lower = text.lower().strip()
        key = text_lower[:100] if len(text_lower) > 100 else text_lower
        if key not in seen:
            seen.add(key)
            unique_contents.append(text)
    
    context = "\n---\n".join(unique_contents)
    return context[:2000]


def expand_query(question: str) -> str:
    """توسيع ذكي للسؤال"""
    q = question.strip().lower()
    original = question.strip()
    
    if any(w in q for w in ["السعر", "سعر", "كم", "دينار", "تكلفة", "ثمن"]):
        return f"{original} باقات الإنترنت الأسعار FTTH WiFi دينار عراقي فايبر وايرلس"
    
    if any(w in q for w in ["باقة", "اشتراك", "الباقات", "باقات", "خطة"]):
        return f"{original} باقات الإنترنت FTTH WiFi المنصة فايبر Star Sun Neptune Galaxy Star"
    
    if any(w in q for w in ["تغطية", "منطقة", "أين", "فرع", "مكان"]):
        return f"{original} تغطية بغداد ديالى بابل المحافظات فرع موقع"
    
    if any(w in q for w in ["دعم", "مساعده", "24", "خدمة", "اتصال"]):
        return f"{original} دعم فني 24/7 خدمة العملاء تواصل"
    
    return original


def is_arabic_text(text: str) -> bool:
    """فحص إذا كان النص عربيًا"""
    if not text or not text.strip():
        return False
    
    arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06FF' or char in '،؛؟')
    total_chars = len([c for c in text if c.isalpha() or c in '،؛؟'])
    
    if total_chars == 0:
        return False
    
    arabic_ratio = arabic_chars / total_chars if total_chars > 0 else 0
    return arabic_ratio >= 0.3


def validate_answer(answer: str, context: str) -> tuple[bool, str]:
    """التحقق من صحة الإجابة"""
    if not answer or len(answer.strip()) < 10:
        return False, "الإجابة قصيرة جدًا"
    
    if not is_arabic_text(answer):
        return False, "الإجابة ليست بالعربية"
    
    words = answer.split()
    if len(set(words)) < len(words) * 0.3:
        return False, "الإجابة تحتوي على تكرار مفرط"
    
    return True, "صحيحة"


# سلسلة RAG
rag_chain = (
    {"context": retriever | filter_and_deduplicate_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def get_answer(question: str, max_retries: int = 2) -> str:
    """الحصول على إجابة دقيقة"""
    if not question or not question.strip():
        return "مرحباً! 🌞 كيف يمكنني مساعدتك اليوم؟ اسألني عن باقات الإنترنت، الأسعار، التغطية، أو أي معلومات عن شمس تيليكوم."

    clean_q = question.strip()
    logger.info(f"معالجة السؤال: '{clean_q}'")

    if not is_arabic_text(clean_q):
        logger.warning(f"السؤال قد لا يكون بالعربية: '{clean_q}'")

    expanded = expand_query(clean_q)
    if expanded != clean_q:
        logger.debug(f"السؤال الموسّع: {expanded}")

    context_used = ""
    
    try:
        docs = retriever.invoke(expanded)
        logger.info(f"تم استرجاع {len(docs)} مستند(ات)")

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
            
            if not docs:
                docs = fallback_retriever.invoke(clean_q)
                logger.info(f"تم الاسترجاع باستخدام retriever احتياطي: {len(docs)} مستند(ات)")

        context_used = filter_and_deduplicate_docs(docs)
        
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = rag_chain.invoke(clean_q)
                response = response.strip()
                
                is_valid, validation_msg = validate_answer(response, context_used)
                if is_valid:
                    logger.info(f"تم التحقق من صحة الإجابة (المحاولة {attempt + 1})")
                    break
                else:
                    logger.warning(f"الإجابة غير صحيحة: {validation_msg} (المحاولة {attempt + 1})")
                    if attempt < max_retries:
                        continue
                    else:
                        response = "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية.\n\nهل تريد معرفة معلومات عن باقاتنا أو خدماتنا؟ 😊"
            except Exception as e:
                logger.error(f"خطأ في المحاولة {attempt + 1}: {e}")
                if attempt < max_retries:
                    continue
                else:
                    raise
        
        if not response:
            response = "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية.\n\nهل تريد معرفة معلومات عن باقاتنا أو خدماتنا؟ 😊"
        
        import re
        response = re.sub(r'^(Answer|Response|Reply):\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        if not is_arabic_text(response):
            logger.warning("الإجابة قد تحتوي على نص غير عربي، سيتم إعادة المحاولة...")
            try:
                response = rag_chain.invoke(f"{clean_q}\n\nتأكد من الإجابة باللغة العربية فقط.")
                response = response.strip()
            except:
                pass
        
        return response

    except Exception as e:
        logger.error(f"خطأ في معالجة السؤال: {e}", exc_info=True)
        return "عذرًا، حدث خطأ تقني. يرجى المحاولة لاحقًا أو الاتصال بنا على 6449."

