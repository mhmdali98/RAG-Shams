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

# 2. إثراء المستند بـ metadata
print("🏷️  جاري إثراء المستند بـ metadata...")
full_text = documents[0].page_content

sections = re.split(r"(?=\n===\s.+?\s===)", full_text)
enhanced_docs = []

for section in sections:
    if not section.strip():
        continue

    header_match = re.search(r"===\s*(.+?)\s*===", section)
    section_name = header_match.group(1).strip() if header_match else "عام"

    if "أخبار ومقالات" in section_name or "شاركنا في فعالية" in section:
        lines = section.split("\n")
        brief_section = "\n".join([lines[0], *[line for line in lines[1:4] if line.strip()]])
        content = brief_section
    else:
        content = section

    enhanced_docs.append({
        "page_content": content.strip(),
        "metadata": {"section": section_name}
    })

print(f"✅ تم تقسيم النص إلى {len(enhanced_docs)} قسم")

# 3. تجزئة النص
print("✂️  جاري تجزئة النص...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=80,
    separators=["\n\n=== ", "\n\n---\n\n", "\n\n", "\n", ". ", "، ", " ", ""],
    length_function=len,
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
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 4, "score_threshold": 0.25}
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

