"""
rebuild_vectorstore.py
إعادة بناء قاعدة البيانات المتجهة بإعدادات محسّنة
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import shutil
import os

print("⚙️ جاري إعادة بناء قاعدة البيانات المتجهة بإعدادات محسّنة...")

# حذف قاعدة البيانات القديمة إذا كانت موجودة
if os.path.exists("./chroma_db"):
    print("🗑️  حذف قاعدة البيانات القديمة...")
    shutil.rmtree("./chroma_db")

# 1. تحميل النص
print("📄 جاري تحميل النص...")
loader = TextLoader("shams-info.txt", encoding="utf-8")
documents = loader.load()
print(f"✅ تم تحميل {len(documents)} مستند")

# 2. تجزئة النص بإعدادات محسّنة
print("✂️  جاري تجزئة النص بإعدادات محسّنة...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,        # تقليل حجم القطعة أكثر لتحسين الدقة
    chunk_overlap=80,      # زيادة التداخل لضمان عدم فقدان المعلومات
    separators=["\n\n=== ", "\n\n", "---\n\n", "\n", ". ", "، ", " ", ""],  # فواصل أفضل
    length_function=len,
)

chunks = text_splitter.split_documents(documents)
print(f"✅ تم تجزئة النص إلى {len(chunks)} جزء")

# 3. تهيئة نموذج التضمين
print("🔢 جاري تحميل نموذج التضمين...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# اختبار التضمين
test_vec = embeddings.embed_query("تجربة")
print(f"✅ نموذج التضمين جاهز! طول المتجه: {len(test_vec)}")

# 4. إنشاء قاعدة بيانات متجهة جديدة
print("💾 جاري إنشاء قاعدة البيانات المتجهة...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# حفظ دائم
vectorstore.persist()
print("✅ تم حفظ قاعدة البيانات المتجهة في './chroma_db'")

# 5. اختبار الاسترجاع
print("\n🧪 اختبار استرجاع المعلومات...")
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

test_queries = [
    "ما هي أسعار باقات الإنترنت؟",
    "اشتراكات الكابل الضوئي",
    "باقات WiFi"
]

for query in test_queries:
    print(f"\n❓ السؤال: {query}")
    results = retriever.invoke(query)
    print(f"   📊 تم استرجاع {len(results)} مستند(ات)")
    if results:
        print(f"   📄 أول نتيجة: {results[0].page_content[:100]}...")

print("\n✅ اكتملت إعادة بناء قاعدة البيانات المتجهة بنجاح!")
print("💡 يمكنك الآن إعادة تشغيل التطبيق للاستفادة من التحسينات")

