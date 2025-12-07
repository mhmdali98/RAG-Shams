"""
rebuild_vectorstore.py
إعادة بناء قاعدة البيانات المتجهة
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import shutil
import os
import re

from config import Settings

print("⚙️ جاري إعادة بناء قاعدة البيانات المتجهة بإعدادات محسّنة...")

# حذف قاعدة البيانات القديمة
if Settings.CHROMA_DB_DIR.exists():
    print("🗑️  حذف قاعدة البيانات القديمة...")
    shutil.rmtree(Settings.CHROMA_DB_DIR)

# إنشاء المجلد إذا لم يكن موجوداً
Settings.CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# 1. تحميل النص
print("📄 جاري تحميل النص...")
loader = TextLoader(str(Settings.DATA_FILE), encoding="utf-8")
documents = loader.load()
print(f"✅ تم تحميل {len(documents)} مستند")

# 2. إثراء المستند بـ metadata ذكي
print("🏷️  جاري إثراء المستند بـ metadata ذكي...")
full_text = documents[0].page_content

sections = re.split(r"(?=\n===\s.+?\s===)", full_text)
enhanced_docs = []

# خريطة الأقسام للفلترة الذكية مع تمييز دقيق بين أنواع الباقات
section_mapping = {
    "ملخص تنفيذي": {"section": "ملخص", "category": "عام", "package_type": None, "keywords": ["باقات", "أسعار", "دعم", "تواصل"]},
    "معلومات الشركة": {"section": "معلومات الشركة", "category": "عام", "package_type": None, "keywords": ["شركة", "تاريخ", "تأسيس", "شمس", "تيليكوم"]},
    "المهمة والقيم": {"section": "المهمة والقيم", "category": "عام", "package_type": None, "keywords": ["مهمة", "قيم", "رؤية"]},
    "الباقات - الكابل الضوئي": {"section": "باقات", "category": "باقات", "package_type": "fiber", "keywords": ["فايبر", "FTTH", "كابل ضوئي", "ألياف", "سعر", "باقة", "35", "50", "75", "150"]},
    "الباقات - الوايرلس": {"section": "باقات", "category": "باقات", "package_type": "wireless", "keywords": ["وايرلس", "WiFi", "wireless", "Star", "Sun", "Neptune", "Galaxy"]},
    "باقات خدمة": {"section": "عروض", "category": "عروض", "package_type": None, "keywords": ["منصة", "ترفيه", "بث"]},
    "الخدمات المقدمة": {"section": "خدمات", "category": "خدمات", "package_type": None, "keywords": ["خدمة", "إنترنت", "FTTH", "WiFi"]},
    "مناطق التغطية": {"section": "تغطية", "category": "معلومات", "package_type": None, "keywords": ["تغطية", "بغداد", "ديالى", "بابل", "المسيب", "الإسكندرية", "سدة الهندية", "فرع"]},
    "معلومات التواصل": {"section": "تواصل", "category": "معلومات", "package_type": None, "keywords": ["هاتف", "بريد", "واتساب", "6449", "info@"]},
    "لماذا تختار": {"section": "مزايا", "category": "عام", "package_type": None, "keywords": ["دعم", "24", "أمن", "وصول"]},
    "الدعم الفني": {"section": "دعم", "category": "خدمات", "package_type": None, "keywords": ["دعم", "فني", "24", "مساعدة", "مشاكل"]},
    "الأسئلة الشائعة": {"section": "FAQ", "category": "معلومات", "package_type": None, "keywords": ["سؤال", "جواب", "شائع"]},
    "طرق الدفع": {"section": "دفع", "category": "خدمات", "package_type": None, "keywords": ["دفع", "تجديد", "باقة"]},
    "تجديد الباقات": {"section": "تجديد", "category": "خدمات", "package_type": None, "keywords": ["تجديد", "باقة", "دفع"]},
}

def detect_section_category(section_name: str, content: str) -> dict:
    """اكتشاف فئة القسم والكلمات المفتاحية مع تمييز دقيق بين أنواع الباقات"""
    content_lower = content.lower()
    
    # البحث في خريطة الأقسام
    for key, info in section_mapping.items():
        if key in section_name:
            return info
    
    # اكتشاف تلقائي من المحتوى
    category = "عام"
    package_type = None
    keywords = []
    
    # تمييز دقيق بين باقات الفايبر والوايرلس
    if any(kw in content_lower for kw in ["فايبر", "ftth", "كابل ضوئي", "ألياف"]):
        category = "باقات"
        package_type = "fiber"
        keywords = ["فايبر", "FTTH", "كابل ضوئي", "35", "50", "75", "150"]
    elif any(kw in content_lower for kw in ["وايرلس", "wireless", "wifi", "star", "sun", "neptune", "galaxy"]):
        category = "باقات"
        package_type = "wireless"
        keywords = ["وايرلس", "WiFi", "wireless", "Star", "Sun", "Neptune", "Galaxy"]
    elif any(kw in content_lower for kw in ["باقة", "سعر", "دينار"]):
        category = "باقات"
        keywords = ["باقة", "سعر", "باقات"]
    elif any(kw in content_lower for kw in ["دعم", "فني", "24", "مساعدة"]):
        category = "خدمات"
        keywords = ["دعم", "فني"]
    elif any(kw in content_lower for kw in ["تغطية", "بغداد", "ديالى", "بابل"]):
        category = "معلومات"
        keywords = ["تغطية", "منطقة", "بغداد", "ديالى", "بابل"]
    elif any(kw in content_lower for kw in ["هاتف", "بريد", "واتساب", "6449"]):
        category = "معلومات"
        keywords = ["تواصل", "هاتف"]
    elif any(kw in content_lower for kw in ["شركة", "شمس", "تيليكوم", "من نحن"]):
        category = "عام"
        keywords = ["شركة", "شمس", "تيليكوم"]
    
    return {
        "section": section_name,
        "category": category,
        "package_type": package_type,
        "keywords": keywords
    }

for section in sections:
    if not section.strip():
        continue

    header_match = re.search(r"===\s*(.+?)\s*===", section)
    section_name = header_match.group(1).strip() if header_match else "عام"

    # استخراج metadata من العلامات [القسم: ...]
    metadata_tags = {}
    tag_match = re.search(r'\[القسم:\s*(.+?)\]', section)
    if tag_match:
        metadata_tags["tag_section"] = tag_match.group(1).strip()
    
    type_match = re.search(r'\[النوع:\s*(.+?)\]', section)
    if type_match:
        metadata_tags["tag_type"] = type_match.group(1).strip()
    
    # اكتشاف فئة القسم
    section_info = detect_section_category(section_name, section)
    
    # فلترة الأخبار الطويلة
    if "أخبار ومقالات" in section_name or "شاركنا في فعالية" in section:
        lines = section.split("\n")
        brief_section = "\n".join([lines[0], *[line for line in lines[1:4] if line.strip()]])
        content = brief_section
    else:
        content = section

    # إنشاء metadata شامل مع تمييز دقيق بين أنواع الباقات
    metadata = {
        "section": section_info["section"],
        "category": section_info["category"],
        "keywords": ", ".join(section_info["keywords"]) if section_info["keywords"] else "",
        **metadata_tags
    }
    
    # إضافة package_type إذا كان موجوداً (للتمييز بين الفايبر والوايرلس)
    if "package_type" in section_info and section_info["package_type"]:
        metadata["package_type"] = section_info["package_type"]

    enhanced_docs.append({
        "page_content": content.strip(),
        "metadata": metadata
    })

print(f"✅ تم تقسيم النص إلى {len(enhanced_docs)} قسم مع metadata ذكي")

# 3. تجزئة النص - محسّنة للأسئلة العامة
print("✂️  جاري تجزئة النص...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # حجم أكبر ليشمل قسم الباقات كاملاً
    chunk_overlap=100,
    separators=[
        "\n=== ",           # فصل الأقسام الرئيسية (الأولوية الأولى)
        "\n---\n",          # فصل الأقسام الفرعية
        "\n\n",             # فقرات
        "\n",               # أسطر
        ". ",               # جمل
        "، ",               # فواصل عربية
        " "                 # كلمات
    ],
    length_function=len,
    is_separator_regex=False
)

final_chunks = []
for doc in enhanced_docs:
    chunks = text_splitter.split_text(doc["page_content"])
    for chunk in chunks:
        if chunk.strip():
            final_chunks.append({
                "page_content": chunk.strip(),
                "metadata": doc["metadata"]
            })

print(f"✅ تم توليد {len(final_chunks)} جزءًا جاهزًا للتضمين")

# 4. تهيئة نموذج التضمين
print("🔢 جاري تحميل نموذج التضمين...")
embeddings = HuggingFaceEmbeddings(
    model_name=Settings.EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

test_vec = embeddings.embed_query("تجربة")
print(f"✅ نموذج التضمين جاهز! طول المتجه: {len(test_vec)}")

# 5. إنشاء مستندات LangChain
langchain_docs = [
    Document(page_content=item["page_content"], metadata=item["metadata"])
    for item in final_chunks
]

# 6. إنشاء قاعدة بيانات متجهة
print("💾 جاري إنشاء قاعدة البيانات المتجهة...")
vectorstore = Chroma.from_documents(
    documents=langchain_docs,
    embedding=embeddings,
    persist_directory=str(Settings.CHROMA_DB_DIR)
)

vectorstore.persist()
print(f"✅ تم حفظ قاعدة البيانات المتجهة في '{Settings.CHROMA_DB_DIR}'")

# 7. اختبار الاسترجاع
print("\n🧪 اختبار استرجاع المعلومات...")
# استخدام retriever بدون threshold للاختبار (أكثر موثوقية)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

test_queries = [
    "ما سعر باقة فايبر 75؟",
    "هل لديكم دعم فني 24 ساعة؟",
    "ما هي مناطق التغطية؟",
    "من هم شركاؤكم؟"
]

for query in test_queries:
    print(f"\n❓ السؤال: {query}")
    results = retriever.invoke(query)
    print(f"   📊 تم استرجاع {len(results)} نتيجة")
    if results:
        print(f"   📂 القسم: {results[0].metadata.get('section', 'غير معروف')}")
        print(f"   📄 المحتوى: {results[0].page_content[:120]}...")

print("\n✅ اكتملت إعادة بناء قاعدة البيانات المتجهة بنجاح!")

