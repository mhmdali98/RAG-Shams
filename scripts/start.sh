#!/bin/bash

# سكريبت لبدء المشروع

echo "🚀 بدء تشغيل بوت شركة الشمس تيليكوم..."
echo ""

# التحقق من البيئة الافتراضية
if [ ! -d "env" ]; then
    echo "❌ البيئة الافتراضية غير موجودة!"
    echo "يرجى إنشاء البيئة أولاً: python -m venv env"
    exit 1
fi

# تفعيل البيئة
source env/bin/activate

# التحقق من قاعدة البيانات
if [ ! -d "storage/chroma_db" ] || [ -z "$(ls -A storage/chroma_db 2>/dev/null)" ]; then
    echo "⚠️  قاعدة البيانات غير موجودة!"
    echo "جاري إعادة بناء قاعدة البيانات..."
    python scripts/rebuild_vectorstore.py
fi

echo ""
echo "اختر طريقة التشغيل:"
echo "1) واجهة Gradio (UI)"
echo "2) واجهة API"
read -p "اختر (1 أو 2): " choice

case $choice in
    1)
        echo "🚀 تشغيل واجهة Gradio..."
        python -m src.ui.app
        ;;
    2)
        echo "🚀 تشغيل واجهة API..."
        uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
        ;;
    *)
        echo "❌ اختيار غير صحيح"
        exit 1
        ;;
esac

