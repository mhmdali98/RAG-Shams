"""
rag_engine.py
محرك RAG محسّن للإجابة الدقيقة والاحترافية على أسئلة العملاء
حول شركة الشمس تيليكوم.
"""

import logging
from typing import List, Optional
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
    
    # Retriever ذكي: يجمع بين k عالي وعتبة تشابه
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 8, "score_threshold": 0.25}
    )
    
    logger.info("جاري تهيئة النموذج اللغوي...")
    llm = ChatOllama(model="llama3", temperature=0.05, num_ctx=3072)  # أقل حرارة = أكثر دقة
    
    logger.info("✅ تم تهيئة النظام بنجاح")
    
except Exception as e:
    logger.error(f"❌ خطأ في تهيئة النظام: {str(e)}")
    raise

# قالب تعليمات مركّز على الاستخلاص الدقيق
prompt = ChatPromptTemplate.from_messages([
    ("system", """أنت مساعد ذكي لشركة "الشمس تيليكوم". مهمتك:
1. **استخلاص المعلومات مباشرة** من السياق أدناه — لا تختلق.
2. **نظم الإجابة** باستخدام عناوين فرعية أو نقاط حسب الحاجة.
3. إذا طُلب "الباقات" أو "الأسعار"، **اذكر جميع الباقات المتوفرة** مع:
   - اسم الباقة
   - السعر (بالدينار العراقي)
   - نوع الباقة (فايبر/وايرلس/منصة)
   - وصف موجز
4. إذا لم تجد معلومات، قل: "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية."
5. كن موجزًا، احترافيًا، ووديًا.
6. لا تكرر نفس المعلومة أكثر من مرة.
"""),
    ("human", """**السياق (استخلص منه فقط):**
{context}

**سؤال العميل:**
{input}

**الإجابة (استنادًا حصريًا إلى السياق أعلاه):**""")
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
        content = doc.page_content
        # استبعاد إذا كان يحتوي على كلمات مثل "شاركنا في فعالية" أو "ورشة عمل"
        if any(term in content for term in ["شاركنا في فعالية", "ورشة عمل", "رعاة", "حدث", "مؤتمر"]):
            if len(content) > 300:
                # نبقي فقط جملة موجزة إن وُجدت
                first_line = content.split("\n")[0]
                if "شمس" in first_line or "باقة" in first_line or "سعر" in first_line:
                    filtered.append(first_line)
                continue
        filtered.append(content)
    
    # إزالة التكرار تقريبيًا
    unique_contents = []
    seen = set()
    for text in filtered:
        key = text[:50].strip().lower()
        if key not in seen:
            seen.add(key)
            unique_contents.append(text)
    
    # دمج النص مع حد أقصى ~1500 حرف (لتحسين السرعة وتجنب overflow)
    context = "\n---\n".join(unique_contents)
    return context[:1500]


# سلسلة RAG محسّنة
rag_chain = (
    {"context": retriever | filter_and_deduplicate_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def expand_query(question: str) -> str:
    """توسيع ذكي للسؤال بناءً على أنماط شمس تيليكوم"""
    q = question.strip().lower()
    original = question.strip()
    
    # أنماط البحث الشائعة
    if any(w in q for w in ["السعر", "سعر", "كم", "دينار"]):
        return f"{original} باقات الإنترنت الأسعار FTTH WiFi دينار عراقي"
    
    if any(w in q for w in ["باقة", "اشتراك", "الباقات", "باقات"]):
        return f"{original} باقات الإنترنت FTTH WiFi المنصة فايبر Star Sun Neptune Galaxy Star"
    
    if "تغطية" in q or "منطقة" in q or "أين" in q:
        return f"{original} تغطية بغداد ديالى بابل المحافظات"
    
    if "دعم" in q or "مساعده" in q or "24" in q:
        return f"{original} دعم فني 24/7 خدمة العملاء"
    
    return original


def get_answer(question: str) -> str:
    """
    الحصول على إجابة دقيقة وسريعة
    """
    if not question or not question.strip():
        return "مرحباً! 🌞 كيف يمكنني مساعدتك اليوم؟ اسألني عن باقات الإنترنت، الأسعار، التغطية، أو أي معلومات عن شمس تيليكوم."

    clean_q = question.strip()
    logger.info(f"معالجة السؤال: '{clean_q}'")

    # توسيع الاستعلام
    expanded = expand_query(clean_q)
    if expanded != clean_q:
        logger.debug(f"السؤال الموسّع: {expanded}")

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

        # تنفيذ السلسلة
        response = rag_chain.invoke(clean_q)
        
        # تنظيف الإجابة
        response = response.strip()
        if not response or "لا أملك" in response or "غير متوفر" in response:
            response = "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية.\n\nهل تريد معرفة معلومات عن باقاتنا أو خدماتنا؟ 😊"
        
        return response

    except Exception as e:
        logger.error(f"خطأ في معالجة السؤال: {e}")
        return "عذرًا، حدث خطأ تقني. يرجى المحاولة لاحقًا أو الاتصال بنا على 6449."