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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
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
    
    # Retriever محسّن - بدون threshold للأسئلة المحددة (أكثر موثوقية)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5  # عدد معقول للأسئلة المحددة
        }
    )
    
    # Retriever مع threshold (للتحكم في الجودة)
    threshold_retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": Settings.RETRIEVER_K,
            "score_threshold": Settings.RETRIEVER_SCORE_THRESHOLD
        }
    )
    
    # Retriever للأسئلة العامة (يستخدم k أكبر)
    general_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 12  # عدد أكبر للأسئلة العامة
        }
    )
    
    # Retriever احتياطي (بدون threshold)
    fallback_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
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


def smart_retriever(question: str, is_general: bool = False):
    """Retriever ذكي يفلتر النتائج حسب نوع السؤال مع تمييز دقيق بين أنواع الباقات"""
    question_lower = question.lower()
    
    # اكتشاف نوع السؤال وتحديد القسم المناسب
    filter_metadata = None
    
    # تمييز دقيق بين باقات الفايبر والوايرلس
    is_about_fiber = any(w in question_lower for w in ["فايبر", "ftth", "كابل ضوئي", "ألياف", "fiber"])
    is_about_wireless = any(w in question_lower for w in ["وايرلس", "wireless", "wifi", "star", "sun", "neptune", "galaxy"])
    
    # دعم فني - استخدام section فقط (ChromaDB يدعم حقل واحد فقط)
    if any(w in question_lower for w in ["دعم", "فني", "24", "مساعدة", "مشكلة", "عطل"]):
        filter_metadata = {"section": "دعم"}
        logger.info("🎯 فلترة: قسم الدعم")
    
    # تغطية ومناطق - فلترة صارمة
    elif any(w in question_lower for w in ["تغطية", "منطقة", "محافظة", "فرع", "مكان", "أين", "بغداد", "ديالى", "بابل"]):
        filter_metadata = {"section": "تغطية"}
        logger.info("🎯 فلترة: قسم التغطية (بغداد، ديالى، بابل فقط)")
    
    # باقات الفايبر تحديداً - استخدام package_type إذا كان متوفراً، وإلا section
    elif is_about_fiber and not is_about_wireless:
        # جرب package_type أولاً، إذا فشل استخدم section
        filter_metadata = {"package_type": "fiber"}
        logger.info("🎯 فلترة: باقات الفايبر (FTTH) فقط - package_type")
    
    # باقات الوايرلس تحديداً
    elif is_about_wireless and not is_about_fiber:
        filter_metadata = {"package_type": "wireless"}
        logger.info("🎯 فلترة: باقات الوايرلس (WiFi) فقط - package_type")
    
    # باقات عامة (بدون تحديد النوع)
    elif any(w in question_lower for w in ["باقة", "باقات", "سعر", "أسعار", "اشتراك"]) and not is_about_fiber and not is_about_wireless:
        filter_metadata = {"section": "باقات"}
        logger.info("🎯 فلترة: قسم الباقات (جميع الأنواع)")
    
    # منصة ترفيهية
    elif any(w in question_lower for w in ["منصة", "ترفيه", "بث"]):
        filter_metadata = {"section": "عروض"}
        logger.info("🎯 فلترة: عروض المنصة الترفيهية")
    
    # معلومات الشركة
    elif any(w in question_lower for w in ["اسم", "شركة", "من نحن", "عن الشركة", "شمس", "تيليكوم"]):
        filter_metadata = {"section": "معلومات الشركة"}
        logger.info("🎯 فلترة: معلومات الشركة")
    
    # تواصل
    elif any(w in question_lower for w in ["تواصل", "اتصال", "هاتف", "بريد", "واتساب", "رقم", "6449"]):
        filter_metadata = {"section": "تواصل"}
        logger.info("🎯 فلترة: قسم التواصل")
    
    # دفع وتجديد
    elif any(w in question_lower for w in ["دفع", "دفعة", "تجديد", "تجديد باقة"]):
        filter_metadata = {"section": "دفع"}
        logger.info("🎯 فلترة: قسم الدفع")
    
    # إذا كان هناك فلترة، استخدم retriever مع فلترة
    if filter_metadata:
        try:
            # محاولة استخدام فلترة Chroma (حقل واحد فقط)
            filtered_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": 12 if is_general else 5,
                    "filter": filter_metadata
                }
            )
            logger.info(f"✅ تم إنشاء retriever مع فلترة: {filter_metadata}")
            return filtered_retriever
        except Exception as e:
            logger.warning(f"⚠️ فشل استخدام فلترة Chroma: {e}. استخدام retriever عادي.")
            # إذا فشلت الفلترة (مثل package_type غير موجود)، جرب section
            if "package_type" in filter_metadata:
                try:
                    # جرب section بدلاً من package_type
                    fallback_filter = {"section": "باقات"}
                    filtered_retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={
                            "k": 12 if is_general else 5,
                            "filter": fallback_filter
                        }
                    )
                    logger.info(f"✅ تم استخدام فلترة احتياطية: {fallback_filter}")
                    return filtered_retriever
                except:
                    pass
            # إذا فشلت الفلترة، استخدم retriever عادي
            if is_general:
                return general_retriever
            else:
                return retriever
    
    # إذا لم يكن هناك فلترة محددة، استخدم retriever عادي
    if is_general:
        return general_retriever
    else:
        return retriever

def get_prompt(question: str, previous_question: str = None, previous_answer: str = None) -> ChatPromptTemplate:
    """إرجاع prompt مخصص حسب نوع السؤال مع دعم الأسئلة التتابعية"""
    # استخراج النصوص
    q_text = extract_text_from_message(question)
    prev_q_text = extract_text_from_message(previous_question) if previous_question else None
    prev_a_text = extract_text_from_message(previous_answer) if previous_answer else None
    
    question_lower = q_text.lower().strip()
    is_followup = is_followup_question(q_text, prev_q_text)
    
    # تحقق من الترحيب
    if is_greeting(q_text) or any(w in question_lower for w in ["من انت", "من أنت", "ماذا تفعل", "ماذا يمكنك"]):
        return ChatPromptTemplate.from_messages([
            ("system", """أنت مساعد ذكي لشركة "شمس تيليكوم". قدّم تحية ودية وعرض خدمات بسيط.

عند الترحيب:
- رحّب بشكل مختصر وودود
- اذكر أنك مساعد لشركة شمس تيليكوم
- اسأل كيف يمكنك المساعدة
- لا تذكر باقات أو أسعار إلا إذا طُلب منك

استخدم المعلومات من النص أدناه عند الحاجة:
{context}"""),
            ("human", "{input}")
        ])
    
    # للأسئلة العادية - إجابة مباشرة بدون تحية
    followup_context = ""
    if is_followup and prev_q_text and prev_a_text:
        followup_context = f"""

**ملاحظة مهمة: هذا سؤال تابع للسؤال السابق:**
- السؤال السابق: "{prev_q_text}"
- الإجابة السابقة: "{prev_a_text[:200]}..."

استخدم السياق أدناه للإجابة على السؤال التابع. إذا كان السؤال يشير إلى معلومة من السؤال السابق (مثل "وما سرعتها؟" بعد سؤال عن باقة)، فاستخدم المعلومات من السياق أدناه."""
    
    # بناء system message بدون f-string للجزء الذي يحتوي على {context}
    system_base = """أنت مساعد رسمي لشركة "شمس تيليكوم". مهمتك: **استخراج المعلومات حرفياً من النص أدناه فقط - ممنوع تماماً الاختلاق أو الإضافة**.
{followup_context}
**تعليمات صارمة جداً (يجب اتباعها حرفياً بدون استثناء):**

1. **أجب مباشرة من المعلومات أدناه دون أي تحية أو تعريف بنفسك.**
   - لا تستخدم عبارات مثل "أنا مساعد لشركة شمس تيليكوم"
   - لا تستخدم عبارات مثل "كيف يمكنني مساعدتك؟"
   - لا تستخدم عبارات مثل "مرحباً" أو "أهلاً"
   - ابدأ الإجابة مباشرة بالمعلومة المطلوبة

2. **استخدم المعلومات من النص فقط - ممنوع تماماً اختلاق أي معلومة غير موجودة في النص.**
   - إذا لم توجد المعلومة في النص: قل فقط "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية."
   - **ممنوع تماماً** ذكر أي محافظة أو منطقة غير موجودة في النص (مثل: الكوفة، النجف، البصرة، الناصرية، ميسان، كربلاء)
   - **المناطق الصحيحة الوحيدة** المذكورة في النص: بغداد، ديالى، بابل (بما في ذلك المسيب، الإسكندرية، سدة الهندية)

3. **عند السؤال عن اسم الشركة:**
   - الاسم الصحيح: "شمس تيليكوم" أو "شركة شمس تيليكوم"
   - إذا كان السؤال "ما اسم الشركة؟" أو "من أنتم؟" أو "من نحن؟": اذكر الاسم مباشرة من النص

4. **عند السؤال عن الباقات والأسعار:**
   - **تمييز دقيق بين أنواع الباقات:**
     * عند السؤال عن "باقات الفايبر" أو "باقات FTTH": اذكر فقط فايبر 35، فايبر 50، فايبر 75، فايبر 150
     * عند السؤال عن "باقات الوايرلس" أو "باقات WiFi": اذكر فقط Star، Sun، Neptune، Galaxy Star
     * **ممنوع تماماً** ذكر باقات فايبر عند السؤال عن وايرلس والعكس
   - **اجمع كل الباقات** من جميع الأقسام في النص
   - اذكر **كل باقة موجودة** مع اسمها وسعرها الدقيق:
     * باقات الفايبر (FTTH): فايبر 35، فايبر 50، فايبر 75، فايبر 150
     * باقات الوايرلس (WiFi): Star، Sun، Neptune، Galaxy Star
     * عروض المنصة الترفيهية: شهر واحد، ثلاثة أشهر، ستة أشهر (هذه عروض وليست باقات)
   - لا تتجاهل أي باقة حتى لو كانت في فقرة مختلفة
   - انسخ الأسعار **حرفياً** من النص
   - **مهم**: المنصة هي "عروض" وليست "باقة" - استخدم "عروض المنصة" أو "باقات المنصة الترفيهية"

5. **عند السؤال عن مناطق التغطية:**
   - **المناطق الصحيحة الوحيدة**: بغداد، ديالى، بابل (بما في ذلك المسيب، الإسكندرية، سدة الهندية)
   - **ممنوع تماماً** ذكر أي محافظة أو منطقة أخرى غير موجودة في النص
   - إذا لم توجد معلومات عن منطقة معينة في النص، لا تذكرها

6. **عند السؤال عن أي معلومة محددة:**
   - أجب مباشرة بدون جمل ترحيبية أو تعريف
   - إذا وُجدت في النص: انسخها أو لخصها بأقل تغيير ممكن
   - إذا لم توجد: قل فقط "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية."

7. **عند السؤال عن معلومات التواصل:**
   - اذكر التفاصيل فقط: الهاتف (6449)، البريد (info@shams-tele.com)، الواتساب، الفروع
   - لا تذكر باقات أو أسعار
   - أجب مباشرة بدون جمل ترحيبية

8. **ممنوع تماماً:**
   - اختلاق أي رقم، سعر، اسم باقة، أو سرعة غير موجودة في النص
   - ذكر محافظات أو مناطق غير موجودة في النص (الكوفة، النجف، البصرة، الناصرية، ميسان، كربلاء، كركوك، سليمانية، أربيل، دهوك)
   - خلط باقات الفايبر مع باقات الوايرلس
   - إضافة جمل مثل "أنا مساعد لشركة شمس تيليكوم"
   - إضافة جمل مثل "كيف يمكنك المساعدة؟"
   - إضافة جمل مثل "لقد جمعنا من النص المرجعي"
   - إضافة جمل مثل "(لا تذكر باقات أو أسعار إلا إذا طُلب منك)"
   - إضافة جمل مثل "نأمل أن هذه المعلومات كانت مفيدة" أو "لا تتردد في الاستفسار"
   - إضافة دعوات للتواصل إلا إذا كان السؤال عن التواصل

9. **أسلوب الإجابة:**
   - مباشر وواضح
   - بدون أي جمل ترحيبية أو تعريف
   - استخدم نقاط أو قوائم عند الحاجة
   - عند ذكر الباقات، نظمها بشكل واضح حسب النوع (فايبر منفصلة عن وايرلس)

**النص المرجعي (استخدمه فقط كمصدر للمعلومات - ممنوع الإضافة أو الاختلاق):**
{context}"""
    
    # استبدال {followup_context} في system_base
    system_message = system_base.replace("{followup_context}", followup_context)
    
    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", """**السياق:**
{context}

**السؤال:**
{input}

**الإجابة (ابدأ مباشرة بالمعلومة بدون تحية أو تعريف):**""")
    ])


def filter_and_deduplicate_docs(docs, question: str = "") -> str:
    """تصفية المستندات وإزالة التكرار - مع أولوية ذكية وتمييز دقيق بين أنواع الباقات"""
    if not docs:
        return "لا توجد معلومات كافية."
    
    question_lower = question.lower() if question else ""
    
    # تحديد نوع السؤال لتحديد الأولوية
    is_about_support = any(w in question_lower for w in ["دعم", "فني", "24", "مساعدة", "مساعدة"])
    is_about_coverage = any(w in question_lower for w in ["تغطية", "منطقة", "محافظة", "فرع", "مكان", "أين"])
    is_about_contact = any(w in question_lower for w in ["تواصل", "اتصال", "هاتف", "بريد", "واتساب", "رقم"])
    is_about_packages = any(w in question_lower for w in ["باقة", "باقات", "سعر", "أسعار", "اشتراك"])
    is_about_payment = any(w in question_lower for w in ["دفع", "دفعة", "تجديد"])
    is_about_company = any(w in question_lower for w in ["اسم", "شركة", "من نحن", "عن الشركة", "شمس", "تيليكوم"])
    
    # تمييز دقيق بين باقات الفايبر والوايرلس
    is_about_fiber = any(w in question_lower for w in ["فايبر", "ftth", "كابل ضوئي", "ألياف", "fiber"])
    is_about_wireless = any(w in question_lower for w in ["وايرلس", "wireless", "wifi", "star", "sun", "neptune", "galaxy"])
    
    # المناطق الصحيحة فقط (منع الهلوسة)
    valid_regions = ["بغداد", "ديالى", "بابل", "المسيب", "الإسكندرية", "سدة الهندية"]
    invalid_regions = ["كوفة", "النجف", "بصرة", "البصرة", "الناصرية", "ميسان", "كربلاء", "كركوك", "سليمانية", "أربيل", "دهوك"]
    
    # تصنيف المستندات حسب الأولوية والعلاقة بالسؤال
    exact_match = []  # تطابق دقيق مع السؤال
    high_priority = []  # باقات، أسعار، FAQ
    medium_priority = []  # خدمات، معلومات عامة
    low_priority = []  # أخبار، أحداث
    irrelevant = []  # غير ذي صلة
    
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
        
        content_lower = content.lower()
        
        # فلترة صارمة: إزالة أي محتوى يذكر مناطق غير صحيحة (منع الهلوسة)
        if is_about_coverage:
            # إذا كان السؤال عن التغطية، رفض أي محتوى يذكر مناطق غير موجودة في البيانات
            has_invalid_region = any(region in content_lower for region in invalid_regions)
            if has_invalid_region:
                logger.warning(f"⚠️ تم رفض مستند يحتوي على مناطق غير صحيحة: {content[:100]}...")
                irrelevant.append(content)
                continue
        
        # فلترة الأخبار والأحداث الطويلة
        if any(term in content_lower for term in ["شاركنا في فعالية", "ورشة عمل", "رعاة", "حدث", "مؤتمر", "معرض", "بودكاست"]):
            if len(content) > 200:
                continue
        
        # معلومات الشركة
        if is_about_company:
            if any(kw in content_lower for kw in ["شمس تيليكوم", "شركة", "من نحن", "معلومات الشركة", "تأسيس"]):
                exact_match.append(content)
                continue
            elif "دفع" in content_lower or "باقة" in content_lower:
                irrelevant.append(content)
                continue
        
        # تحديد الأولوية حسب السؤال
        if is_about_support:
            if any(kw in content_lower for kw in ["دعم", "فني", "24", "ساعة", "6449"]):
                exact_match.append(content)
                continue
            elif "دفع" in content_lower or "تجديد" in content_lower:
                irrelevant.append(content)
                continue
        
        if is_about_coverage:
            # فلترة صارمة: فقط المحافظات الصحيحة
            has_valid_region = any(region in content_lower for region in valid_regions)
            if has_valid_region and any(kw in content_lower for kw in ["تغطية", "منطقة", "بغداد", "ديالى", "بابل", "فرع"]):
                exact_match.append(content)
                continue
            elif "دفع" in content_lower or "باقة" in content_lower:
                irrelevant.append(content)
                continue
        
        if is_about_contact:
            if any(kw in content_lower for kw in ["تواصل", "هاتف", "بريد", "واتساب", "6449", "info@"]):
                exact_match.append(content)
                continue
            elif "دفع" in content_lower or "باقة" in content_lower:
                irrelevant.append(content)
                continue
        
        if is_about_payment:
            if any(kw in content_lower for kw in ["دفع", "تجديد", "باقة"]):
                exact_match.append(content)
                continue
        
        # تمييز دقيق بين باقات الفايبر والوايرلس
        if is_about_packages:
            # إذا كان السؤال عن الفايبر فقط
            if is_about_fiber and not is_about_wireless:
                # رفض أي محتوى عن الوايرلس
                if any(w in content_lower for w in ["star", "sun", "neptune", "galaxy", "وايرلس", "wireless", "wifi"]):
                    irrelevant.append(content)
                    continue
                # قبول فقط محتوى الفايبر
                if any(kw in content_lower for kw in ["فايبر", "ftth", "كابل ضوئي", "ألياف"]):
                    exact_match.append(content)
                    continue
            
            # إذا كان السؤال عن الوايرلس فقط
            elif is_about_wireless and not is_about_fiber:
                # رفض أي محتوى عن الفايبر
                if any(w in content_lower for w in ["فايبر", "ftth", "كابل ضوئي", "ألياف"]):
                    irrelevant.append(content)
                    continue
                # قبول فقط محتوى الوايرلس
                if any(kw in content_lower for kw in ["star", "sun", "neptune", "galaxy", "وايرلس", "wireless", "wifi"]):
                    exact_match.append(content)
                    continue
        
        # تصنيف عام
        if any(keyword in content_lower for keyword in ["باقة", "سعر", "دينار", "faq", "أسئلة شائعة", "فايبر", "وايرلس", "star", "sun", "neptune", "galaxy", "منصة"]):
            high_priority.append(content)
        elif any(keyword in content_lower for keyword in ["خدمة", "تغطية", "تواصل", "دعم", "فرع", "هاتف", "بريد"]):
            medium_priority.append(content)
        else:
            low_priority.append(content)
    
    # إزالة التكرار
    def deduplicate(texts):
        unique = []
        seen = set()
        for text in texts:
            text_lower = text.lower().strip()
            key = text_lower[:150] if len(text_lower) > 150 else text_lower
            if key not in seen:
                seen.add(key)
                unique.append(text)
        return unique
    
    exact_match = deduplicate(exact_match)
    high_priority = deduplicate(high_priority)
    medium_priority = deduplicate(medium_priority)
    low_priority = deduplicate(low_priority)
    
    # دمج مع الأولوية: exact_match أولاً، ثم high_priority، إلخ
    all_docs = exact_match + high_priority + medium_priority + low_priority[:2]
    
    if not all_docs:
        all_docs = [doc.page_content.strip() for doc in docs[:3] if doc.page_content.strip()]
    
    context = "\n---\n".join(all_docs)
    # تقليل طول السياق لتحسين السرعة (1500 حرف كافٍ لمعظم الأسئلة)
    return context[:1500]


def extract_text_from_message(message) -> str:
    """استخراج النص من message (dict, list, أو string)"""
    if not message:
        return ""
    
    if isinstance(message, str):
        return message
    elif isinstance(message, dict):
        # Gradio format: {'text': '...', 'type': 'text'}
        if 'text' in message:
            return str(message['text'])
        elif 'content' in message:
            return str(message['content'])
        else:
            return str(message)
    elif isinstance(message, list):
        if len(message) > 0:
            # إذا كان list من strings
            if isinstance(message[0], str):
                return message[0]
            # إذا كان list من dicts
            elif isinstance(message[0], dict):
                return extract_text_from_message(message[0])
            else:
                return str(message[0])
        return ""
    else:
        return str(message)


def is_followup_question(question: str, previous_question: str = None) -> bool:
    """فحص إذا كان السؤال تابعاً للسؤال السابق"""
    if not previous_question:
        return False
    
    # استخراج النص من previous_question (قد يكون dict, list, أو string)
    prev_text = extract_text_from_message(previous_question)
    if not prev_text:
        return False
    
    q_lower = question.lower().strip()
    prev_lower = prev_text.lower().strip()
    
    # كلمات تشير إلى سؤال تابع
    followup_indicators = [
        "و", "وأيضاً", "وكذلك", "وما", "وهل", "وكم",
        "ما", "ماذا", "كيف", "أين", "متى", "لماذا",
        "سرعة", "سعر", "تكلفة", "ثمن", "مميزات", "خصائص",
        "تفاصيل", "معلومات", "أكثر", "أيضاً", "كذلك"
    ]
    
    # إذا كان السؤال قصير جداً (< 10 كلمات) ويحتوي على مؤشرات تابع
    if len(q_lower.split()) < 10:
        if any(indicator in q_lower for indicator in followup_indicators):
            # تحقق من وجود كلمات مشتركة مع السؤال السابق
            prev_words = set(prev_lower.split())
            q_words = set(q_lower.split())
            common_words = prev_words.intersection(q_words)
            
            # إذا كان هناك كلمات مشتركة (مثل "باقة", "فايبر", "75")
            if len(common_words) > 0:
                return True
    
    return False


def expand_query(question: str, previous_question: str = None, previous_answer: str = None) -> str:
    """توسيع ذكي للسؤال - محسّن لتحسين الاسترجاع مع تمييز دقيق بين أنواع الباقات"""
    q = question.strip().lower()
    original = question.strip()
    
    # إذا كان سؤال تابع، دمج مع السؤال السابق
    if is_followup_question(original, previous_question):
        logger.info(f"🔗 سؤال تابع: '{original}' (بعد: '{previous_question}')")
        # دمج السؤال الحالي مع السؤال السابق
        combined = f"{previous_question} {original}"
        original = combined
        q = combined.lower()
    
    # تمييز دقيق بين باقات الفايبر والوايرلس
    is_about_fiber = any(w in q for w in ["فايبر", "ftth", "كابل ضوئي", "ألياف", "fiber"])
    is_about_wireless = any(w in q for w in ["وايرلس", "wireless", "wifi", "star", "sun", "neptune", "galaxy"])
    
    # توسيع أكثر تحديداً مع تمييز دقيق
    if any(w in q for w in ["السعر", "سعر", "كم", "دينار", "تكلفة", "ثمن", "أسعار"]):
        if is_about_fiber and not is_about_wireless:
            return f"{original} باقات الفايبر FTTH كابل ضوئي أسعار دينار عراقي فايبر 35 50 75 150"
        elif is_about_wireless and not is_about_fiber:
            return f"{original} باقات الوايرلس WiFi wireless أسعار دينار عراقي Star Sun Neptune Galaxy Star"
        else:
            return f"{original} باقات الإنترنت الأسعار FTTH WiFi دينار عراقي فايبر وايرلس Star Sun Neptune"
    
    if any(w in q for w in ["باقة", "اشتراك", "الباقات", "باقات", "خطة", "جميع"]):
        if is_about_fiber and not is_about_wireless:
            return f"{original} باقات الفايبر FTTH كابل ضوئي فايبر 35 50 75 150"
        elif is_about_wireless and not is_about_fiber:
            return f"{original} باقات الوايرلس WiFi wireless Star Sun Neptune Galaxy Star"
        else:
            return f"{original} باقات الإنترنت FTTH WiFi المنصة فايبر 35 50 75 150 Star Sun Neptune Galaxy Star"
    
    if any(w in q for w in ["تغطية", "منطقة", "أين", "فرع", "مكان", "مناطق", "محافظة"]):
        # توسيع محدد للمناطق الصحيحة فقط
        return f"{original} تغطية بغداد ديالى بابل المسيب الإسكندرية سدة الهندية فرع موقع"
    
    if any(w in q for w in ["دعم", "مساعده", "24", "خدمة", "اتصال", "فني"]):
        return f"{original} دعم فني 24/7 خدمة العملاء تواصل هاتف 6449"
    
    if any(w in q for w in ["اسم", "شركة", "من نحن", "عن الشركة", "شمس", "تيليكوم"]):
        return f"{original} شركة شمس تيليكوم معلومات الشركة من نحن تأسيس"
    
    if any(w in q for w in ["خدمة", "خدمات", "ماذا", "ما هي"]):
        return f"{original} خدمات الإنترنت FTTH WiFi مشاريع بلو سيركل"
    
    if any(w in q for w in ["شركاء", "شريك", "أصدقاء"]):
        return f"{original} شركاء أصدقاء تبادل اسوار تازة المنصة"
    
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
    """التحقق من صحة الإجابة - مع فحص صارم ضد الهلوسة (مناطق، باقات، اسم الشركة)"""
    if not answer or len(answer.strip()) < 10:
        return False, "الإجابة قصيرة جدًا"
    
    if not is_arabic_text(answer):
        return False, "الإجابة ليست بالعربية"
    
    # فحص التكرار المفرط
    words = answer.split()
    if len(set(words)) < len(words) * 0.3:
        return False, "الإجابة تحتوي على تكرار مفرط"
    
    answer_lower = answer.lower()
    context_lower = context.lower()
    
    # === فحص صارم: المناطق غير الصحيحة ===
    invalid_regions = ["كوفة", "النجف", "بصرة", "البصرة", "الناصرية", "ميسان", "كربلاء", "كركوك", "سليمانية", "أربيل", "دهوك"]
    valid_regions = ["بغداد", "ديالى", "بابل", "المسيب", "الإسكندرية", "سدة الهندية"]
    
    # إذا كانت الإجابة تذكر مناطق غير صحيحة
    for invalid_region in invalid_regions:
        if invalid_region in answer_lower and invalid_region not in context_lower:
            logger.error(f"❌ تم اكتشاف منطقة غير صحيحة في الإجابة: {invalid_region}")
            return False, f"إجابة تحتوي على منطقة غير موجودة في البيانات: {invalid_region}"
    
    # === فحص: اسم الشركة ===
    # إذا كانت الإجابة تقول "لا توجد معلومات" أو "لا أعرف" عند السؤال عن اسم الشركة
    if "شمس" in context_lower or "تيليكوم" in context_lower:
        # إذا كان السياق يحتوي على اسم الشركة لكن الإجابة تقول "لا توجد معلومات"
        if any(phrase in answer_lower for phrase in ["لا توجد معلومات", "لا أعرف", "لا يوجد", "غير متوفر"]):
            if "شمس" not in answer_lower and "تيليكوم" not in answer_lower:
                logger.warning("⚠️ الإجابة تقول 'لا توجد معلومات' رغم وجود اسم الشركة في السياق")
                # لا نرفضها تماماً، لكن نسجل تحذير
    
    # === فحص صارم: الباقات ===
    # قائمة الباقات الصحيحة من السياق
    valid_fiber_packages = []
    valid_wireless_packages = []
    
    if "فايبر 35" in context_lower or "فايبر35" in context_lower:
        valid_fiber_packages.append("فايبر 35")
    if "فايبر 50" in context_lower or "فايبر50" in context_lower:
        valid_fiber_packages.append("فايبر 50")
    if "فايبر 75" in context_lower or "فايبر75" in context_lower:
        valid_fiber_packages.append("فايبر 75")
    if "فايبر 150" in context_lower or "فايبر150" in context_lower:
        valid_fiber_packages.append("فايبر 150")
    
    if "star" in context_lower and "galaxy star" not in context_lower:
        valid_wireless_packages.append("star")
    if "sun" in context_lower:
        valid_wireless_packages.append("sun")
    if "neptune" in context_lower:
        valid_wireless_packages.append("neptune")
    if "galaxy star" in context_lower:
        valid_wireless_packages.append("galaxy star")
    
    # فحص الباقات المشبوهة (مثل "فايبر 100" أو "فايبر 500" التي لا توجد في البيانات)
    suspicious_patterns = [
        "فايبر 100", "فايبر100", "100 mbps", "100mbps",
        "فايبر 500", "فايبر500", "500 mbps", "500mbps",
        "فايبر 1 gbps", "1gbps", "1000 mbps"
    ]
    
    for pattern in suspicious_patterns:
        if pattern in answer_lower and pattern not in context_lower:
            logger.error(f"❌ تم اكتشاف باقة مشبوهة في الإجابة: {pattern}")
            return False, f"إجابة تحتوي على باقة غير موجودة في البيانات: {pattern}"
    
    # === فحص: خلط بين باقات الفايبر والوايرلس ===
    # إذا كان السؤال عن الوايرلس لكن الإجابة تذكر فايبر
    is_about_wireless_question = any(w in answer_lower for w in ["وايرلس", "wireless", "wifi"])
    has_fiber_in_answer = any(w in answer_lower for w in ["فايبر", "ftth", "كابل ضوئي"])
    has_wireless_in_answer = any(w in answer_lower for w in ["star", "sun", "neptune", "galaxy"])
    
    # إذا كان السؤال عن الوايرلس فقط لكن الإجابة تذكر فايبر
    if is_about_wireless_question and has_fiber_in_answer and not has_wireless_in_answer:
        logger.error("❌ الإجابة تخلط بين باقات الفايبر والوايرلس")
        return False, "إجابة تخلط بين باقات الفايبر والوايرلس"
    
    # === فحص الأسعار المشبوهة ===
    import re
    prices_in_answer = re.findall(r'(\d{1,3}(?:,\d{3})*)\s*دينار', answer)
    prices_in_context = re.findall(r'(\d{1,3}(?:,\d{3})*)\s*دينار', context)
    
    # تحويل إلى أرقام للتحقق
    def parse_price(price_str):
        return int(price_str.replace(',', '').replace('،', ''))
    
    context_prices = set()
    for price_str in prices_in_context:
        try:
            context_prices.add(parse_price(price_str))
        except:
            pass
    
    for price_str in prices_in_answer:
        try:
            price_num = parse_price(price_str)
            # إذا كان السعر كبير جداً (أكثر من 200,000) وغير موجود في السياق، فهو مشبوه
            if price_num > 200000 and price_num not in context_prices:
                logger.warning(f"⚠️ تم اكتشاف سعر مشبوه في الإجابة: {price_num}")
                return False, f"إجابة تحتوي على سعر غير موجود في البيانات: {price_num}"
        except:
            pass
    
    return True, "صحيحة"


def get_rag_chain(question: str, previous_question: str = None, previous_answer: str = None):
    """إرجاع RAG chain ديناميكي حسب نوع السؤال مع دعم الأسئلة التتابعية"""
    # استخراج النصوص من previous_question و previous_answer
    prev_q_text = extract_text_from_message(previous_question) if previous_question else None
    prev_a_text = extract_text_from_message(previous_answer) if previous_answer else None
    
    # الحصول على prompt مخصص
    custom_prompt = get_prompt(question, prev_q_text, prev_a_text)
    
    # استخدام retriever ذكي حسب نوع السؤال
    # للأسئلة التتابعية، استخدم السؤال الأصلي للاسترجاع
    search_question = prev_q_text if is_followup_question(question, prev_q_text) else question
    # استخراج النص من search_question إذا كان dict
    search_question = extract_text_from_message(search_question)
    is_general = is_general_question(search_question)
    smart_ret = smart_retriever(search_question, is_general)
    
    # دالة لتصفية المستندات مع السؤال
    def format_with_question(docs):
        return filter_and_deduplicate_docs(docs, search_question)
    
    # إنشاء chain ديناميكي
    chain = (
        {"context": smart_ret | RunnableLambda(format_with_question), "input": RunnablePassthrough()}
        | custom_prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def is_general_question(question: str) -> bool:
    """فحص إذا كان السؤال عاماً يتطلب معلومات من مصادر متعددة"""
    # استخراج النص إذا كان dict أو list
    q_text = extract_text_from_message(question)
    q_lower = q_text.lower().strip()
    general_keywords = [
        "جميع", "كل", "ما هي", "ما أسعار", "ما الباقات", 
        "جميع الباقات", "جميع الأسعار", "ما هي الباقات",
        "ما أسعار الباقات", "ما هي جميع", "قائمة", "عرض"
    ]
    return any(keyword in q_lower for keyword in general_keywords)


def is_greeting(question: str) -> bool:
    """فحص إذا كان السؤال ترحيباً"""
    q_lower = question.lower().strip()
    greetings = [
        "مرحبا", "مرحباً", "السلام عليكم", "أهلا", "أهلاً",
        "hello", "hi", "صباح الخير", "مساء الخير",
        "كيف حالك", "كيفك", "شلونك"
    ]
    return any(greeting in q_lower for greeting in greetings)


def get_answer(question: str, previous_question: str = None, previous_answer: str = None, max_retries: int = 2) -> str:
    """الحصول على إجابة دقيقة"""
    if not question or not question.strip():
        return "مرحباً! 🌞 أنا مساعدك الذكي لشركة شمس تيليكوم. كيف يمكنني مساعدتك اليوم؟ يمكنك سؤالي عن باقات الإنترنت، الأسعار، التغطية، أو أي معلومات عن خدماتنا."

    clean_q = question.strip()
    logger.info(f"معالجة السؤال: '{clean_q}'")

    # معالجة الترحيب
    if is_greeting(clean_q):
        return "مرحباً! 🌞 أنا مساعدك الذكي لشركة شمس تيليكوم. كيف يمكنني مساعدتك اليوم؟ يمكنك سؤالي عن باقات الإنترنت، الأسعار، التغطية، الدعم الفني، أو أي معلومات عن خدماتنا."

    if not is_arabic_text(clean_q):
        logger.warning(f"السؤال قد لا يكون بالعربية: '{clean_q}'")

    expanded = expand_query(clean_q, previous_question, previous_answer)
    if expanded != clean_q:
        logger.debug(f"السؤال الموسّع: {expanded}")

    context_used = ""
    
    try:
        # استخدام retriever ذكي مع فلترة حسب نوع السؤال
        # ملاحظة: retriever سيُستخدم في get_rag_chain، لكن نحتاج أيضاً لاسترجاع docs هنا للتحقق
        is_general = is_general_question(clean_q)
        smart_ret = smart_retriever(clean_q, is_general)
        
        if is_general:
            logger.info("🔍 سؤال عام - استخدام retriever ذكي مع فلترة")
            docs = smart_ret.invoke(expanded)
            if not docs or len(docs) < 3:
                # إذا لم نحصل على نتائج كافية، جرب بدون توسيع
                docs = smart_ret.invoke(clean_q)
                logger.info(f"تم الاسترجاع بدون توسيع: {len(docs)} مستند(ات)")
        else:
            # للأسئلة المحددة، استخدم retriever ذكي مع فلترة
            logger.info("🎯 سؤال محدد - استخدام retriever ذكي مع فلترة")
            docs = smart_ret.invoke(expanded)
            if not docs:
                # إذا لم نحصل على نتائج، جرب بدون توسيع
                docs = smart_ret.invoke(clean_q)
                logger.info(f"تم الاسترجاع بدون توسيع: {len(docs)} مستند(ات)")
        
        logger.info(f"تم استرجاع {len(docs)} مستند(ات)")

        # إذا لم نحصل على نتائج، جرب استراتيجيات احتياطية
        if not docs or len(docs) == 0:
            logger.warning("⚠️ لم يتم استرجاع أي مستندات، جاري المحاولة باستراتيجيات احتياطية...")
            
            # استراتيجية 1: استعلامات بديلة
            fallback_queries = [
                clean_q,
                " ".join([w for w in clean_q.split() if len(w) > 2]),
            ]
            
            # إضافة كلمات مفتاحية حسب نوع السؤال
            if any(w in clean_q.lower() for w in ["باقة", "سعر", "أسعار"]):
                fallback_queries.append("باقات أسعار دينار")
            if any(w in clean_q.lower() for w in ["دعم", "فني", "24"]):
                fallback_queries.append("دعم فني 24 ساعة")
            if any(w in clean_q.lower() for w in ["تغطية", "منطقة", "فرع"]):
                fallback_queries.append("مناطق التغطية بغداد ديالى")
            if any(w in clean_q.lower() for w in ["خدمة", "خدمات"]):
                fallback_queries.append("خدمات الإنترنت FTTH WiFi")
            
            for fq in fallback_queries:
                if is_general_question(clean_q):
                    docs = general_retriever.invoke(fq)
                else:
                    docs = retriever.invoke(fq)
                if docs and len(docs) > 0:
                    logger.info(f"✅ تم الاسترجاع باستخدام الاستعلام الاحتياطي: {fq} ({len(docs)} مستند)")
                    break
            
            # استراتيجية 2: retriever احتياطي
            if not docs or len(docs) == 0:
                logger.warning("⚠️ المحاولة مع retriever احتياطي...")
                docs = fallback_retriever.invoke(clean_q)
                if docs:
                    logger.info(f"✅ تم الاسترجاع باستخدام retriever احتياطي: {len(docs)} مستند(ات)")

        # إذا لم نحصل على أي مستندات بعد كل المحاولات
        if not docs or len(docs) == 0:
            logger.error("❌ لم يتم استرجاع أي مستندات بعد كل المحاولات")
            return "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية.\n\nيمكنك سؤالي عن:\n- باقات الإنترنت (فايبر، وايرلس، المنصة)\n- الأسعار\n- مناطق التغطية\n- الدعم الفني\n- طرق التواصل"
        
        context_used = filter_and_deduplicate_docs(docs, clean_q)
        
        # إذا كان السياق فارغاً أو قصيراً جداً
        if not context_used or len(context_used.strip()) < 20:
            logger.warning("⚠️ السياق المسترجع قصير جداً أو فارغ")
            return "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية.\n\nيمكنك سؤالي عن:\n- باقات الإنترنت (فايبر، وايرلس، المنصة)\n- الأسعار\n- مناطق التغطية\n- الدعم الفني\n- طرق التواصل"
        
        logger.info(f"📄 طول السياق المستخدم: {len(context_used)} حرف")
        
        # الحصول على RAG chain ديناميكي حسب نوع السؤال
        dynamic_rag_chain = get_rag_chain(clean_q, previous_question, previous_answer)
        
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = dynamic_rag_chain.invoke(clean_q)
                response = response.strip()
                
                # إذا كانت الإجابة قصيرة جداً أو فارغة، أعد المحاولة
                if not response or len(response) < 10:
                    logger.warning(f"⚠️ الإجابة قصيرة جداً (المحاولة {attempt + 1})")
                    if attempt < max_retries:
                        continue
                
                is_valid, validation_msg = validate_answer(response, context_used)
                if is_valid:
                    logger.info(f"✅ تم التحقق من صحة الإجابة (المحاولة {attempt + 1})")
                    break
                else:
                    logger.warning(f"⚠️ الإجابة غير صحيحة: {validation_msg} (المحاولة {attempt + 1})")
                    if attempt < max_retries:
                        continue
                    else:
                        # إذا فشل التحقق، لكن الإجابة ليست فارغة، استخدمها
                        if response and len(response) > 20:
                            logger.warning("⚠️ استخدام الإجابة رغم فشل التحقق")
                            break
                        response = "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية.\n\nيمكنك سؤالي عن:\n- باقات الإنترنت (فايبر، وايرلس، المنصة)\n- الأسعار\n- مناطق التغطية\n- الدعم الفني\n- طرق التواصل"
            except Exception as e:
                logger.error(f"❌ خطأ في المحاولة {attempt + 1}: {e}")
                if attempt < max_retries:
                    continue
                else:
                    raise
        
        if not response:
            response = "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية.\n\nهل تريد معرفة معلومات عن باقاتنا أو خدماتنا؟ 😊"
        
        import re
        
        # تنظيف الإجابة من العلامات غير المرغوبة
        response = re.sub(r'^(Answer|Response|Reply):\s*', '', response, flags=re.IGNORECASE)
        response = response.strip()
        
        # إزالة الجمل الترحيبية والدعوات للتواصل غير المطلوبة
        unwanted_phrases = [
            r'^أنا\s+مساعد\s+لشركة\s+شمس\s+تيليكوم[^.]*\.',
            r'^مرحباً[!.]?\s*',
            r'^مرحبا[!.]?\s*',
            r'^مرحبا[!.]?\s*أنا\s+مساعد[^.]*\.',
            r'كيف\s+يمكنك\s+المساعدة[?.]?',
            r'كيف\s+يمكنني\s+المساعدة[?.]?',
            r'لقد\s+جمعنا[^.]*من\s+النص\s+المرجعي[^.]*\.',
            r'لقد\s+جمعت[^.]*من\s+النص\s+المرجعي[^.]*\.',
            r'من\s+النص\s+المرجعي\s+أدناه[^.]*\.',
            r'نرجو\s+الاتصال\s+بنا[^.]*\.',
            r'يمكنك\s+الاتصال\s+بنا[^.]*\.',
            r'للاستفسار[^.]*اتصل\s+بنا[^.]*\.',
            r'لطلب\s+الخدمة[^.]*اتصل\s+بنا[^.]*\.',
            r'للتواصل[^.]*اتصل\s+بنا[^.]*\.',
            r'نأمل\s+أن\s+هذه\s+المعلومات[^.]*\.',
            r'نأمل\s+أن\s+نكون[^.]*\.',
            r'شكراً\s+لثقتك[^.]*\.',
            r'شكرا\s+لثقتك[^.]*\.',
            r'لا\s+تتردد\s+في\s+الاستفسار[^.]*\.',
            r'إذا\s+كان\s+لديك\s+أي\s+سؤال\s+آخر[^.]*\.',
            r'\(لا\s+تذكر\s+باقات[^)]*\)',
            r'\(لا\s+تذكر\s+أسعار[^)]*\)',
        ]
        
        for phrase in unwanted_phrases:
            response = re.sub(phrase, '', response, flags=re.IGNORECASE | re.MULTILINE)
        
        # إزالة الجمل التي تبدأ بـ "أنا مساعد" في منتصف الإجابة
        response = re.sub(r'أنا\s+مساعد\s+لشركة\s+شمس\s+تيليكوم[^.]*\.', '', response, flags=re.IGNORECASE)
        
        # تنظيف المسافات الزائدة
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = response.strip()
        
        # إذا كانت الإجابة فارغة بعد التنظيف، استخدم رسالة افتراضية
        if not response or len(response) < 10:
            response = "عذرًا، لا توجد معلومات كافية حول هذا الموضوع في قاعدة بياناتنا الحالية."
        
        if not is_arabic_text(response):
            logger.warning("الإجابة قد تحتوي على نص غير عربي، سيتم إعادة المحاولة...")
            try:
                retry_chain = get_rag_chain(clean_q, previous_question, previous_answer)
                response = retry_chain.invoke(f"{clean_q}\n\nتأكد من الإجابة باللغة العربية فقط.")
                response = response.strip()
            except:
                pass
        
        return response

    except Exception as e:
        logger.error(f"خطأ في معالجة السؤال: {e}", exc_info=True)
        return "عذرًا، حدث خطأ تقني. يرجى المحاولة لاحقًا أو الاتصال بنا على 6449."

