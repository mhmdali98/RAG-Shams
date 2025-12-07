"""
ملف اختبار شامل لقياس دقة نموذج RAG
يحتوي على 30 سؤال مع الإجابات المتوقعة
"""

import sys
from pathlib import Path
import json
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core import get_answer
import logging

logging.basicConfig(level=logging.WARNING)  # تقليل الـ logging أثناء الاختبار

# قائمة الأسئلة مع الإجابات المتوقعة (Ground Truth)
TEST_QUESTIONS = [
    # === أسئلة عن الباقات ===
    {
        "question": "ما هي جميع باقات الإنترنت المتوفرة؟",
        "expected_keywords": ["فايبر 35", "فايبر 50", "فايبر 75", "فايبر 150", "Star", "Sun", "Neptune", "Galaxy Star"],
        "category": "باقات"
    },
    {
        "question": "ما سعر باقة فايبر 75؟",
        "expected_keywords": ["65,000", "65000", "دينار"],
        "category": "أسعار"
    },
    {
        "question": "ما باقات الوايرلس؟",
        "expected_keywords": ["Star", "Sun", "Neptune", "Galaxy Star", "وايرلس"],
        "must_not_contain": ["فايبر", "FTTH", "كابل ضوئي"],
        "category": "باقات وايرلس"
    },
    {
        "question": "ما باقات الفايبر؟",
        "expected_keywords": ["فايبر 35", "فايبر 50", "فايبر 75", "فايبر 150"],
        "must_not_contain": ["Star", "Sun", "Neptune", "Galaxy", "وايرلس"],
        "category": "باقات فايبر"
    },
    {
        "question": "كم سعر باقة Sun؟",
        "expected_keywords": ["40,000", "40000", "دينار"],
        "category": "أسعار"
    },
    {
        "question": "ما سعر باقة المنصة لثلاثة أشهر؟",
        "expected_keywords": ["25,000", "25000", "دينار"],
        "category": "أسعار"
    },
    
    # === أسئلة عن مناطق التغطية ===
    {
        "question": "ما هي مناطق التغطية؟",
        "expected_keywords": ["بغداد", "ديالى", "بابل"],
        "must_not_contain": ["كوفة", "النجف", "بصرة", "البصرة", "الناصرية", "ميسان", "كربلاء"],
        "category": "تغطية"
    },
    {
        "question": "هل لديكم فرع في ديالى؟",
        "expected_keywords": ["ديالى", "فرع"],
        "must_not_contain": ["كوفة", "النجف", "بصرة"],
        "category": "تغطية"
    },
    {
        "question": "أين تقع فروعكم؟",
        "expected_keywords": ["بغداد", "ديالى", "بابل"],
        "must_not_contain": ["كوفة", "النجف", "بصرة", "الناصرية"],
        "category": "تغطية"
    },
    {
        "question": "هل تقدمون خدمات في بغداد؟",
        "expected_keywords": ["بغداد"],
        "must_not_contain": ["كوفة", "النجف", "بصرة"],
        "category": "تغطية"
    },
    
    # === أسئلة عن معلومات الشركة ===
    {
        "question": "ما اسم الشركة؟",
        "expected_keywords": ["شمس", "تيليكوم", "شمس تيليكوم"],
        "must_not_contain": ["لا توجد معلومات", "لا أعرف", "غير متوفر"],
        "category": "معلومات الشركة"
    },
    {
        "question": "من نحن؟",
        "expected_keywords": ["شمس", "تيليكوم", "شركة"],
        "must_not_contain": ["لا توجد معلومات"],
        "category": "معلومات الشركة"
    },
    {
        "question": "كم سنة من الخبرة لديكم؟",
        "expected_keywords": ["10", "عشرة", "أكثر من 10"],
        "category": "معلومات الشركة"
    },
    
    # === أسئلة عن الدعم الفني ===
    {
        "question": "هل الدعم الفني متاح 24 ساعة؟",
        "expected_keywords": ["24", "ساعة", "متاح", "دعم"],
        "category": "دعم"
    },
    {
        "question": "ما هو رقم خدمة العملاء؟",
        "expected_keywords": ["6449"],
        "category": "تواصل"
    },
    {
        "question": "كيف أتواصل مع الدعم الفني؟",
        "expected_keywords": ["6449", "هاتف", "واتساب"],
        "category": "دعم"
    },
    
    # === أسئلة عن التواصل ===
    {
        "question": "ما هو البريد الإلكتروني؟",
        "expected_keywords": ["info@shams-tele.com", "shams-tele.com"],
        "category": "تواصل"
    },
    {
        "question": "كيف أتواصل معكم؟",
        "expected_keywords": ["6449", "info@", "واتساب"],
        "category": "تواصل"
    },
    {
        "question": "ما هو رقم الهاتف؟",
        "expected_keywords": ["6449"],
        "category": "تواصل"
    },
    
    # === أسئلة عن الخدمات ===
    {
        "question": "ما هي الخدمات التي تقدمونها؟",
        "expected_keywords": ["FTTH", "وايرلس", "WiFi", "إنترنت"],
        "category": "خدمات"
    },
    {
        "question": "هل تقدمون إنترنت عبر الألياف الضوئية؟",
        "expected_keywords": ["فايبر", "FTTH", "ألياف", "كابل ضوئي"],
        "category": "خدمات"
    },
    {
        "question": "ما هي خدمة بلو سيركل؟",
        "expected_keywords": ["بلو سيركل", "ألياف", "حلول"],
        "category": "خدمات"
    },
    
    # === أسئلة عن الباقات - تفاصيل ===
    {
        "question": "ما سرعة باقة فايبر 150؟",
        "expected_keywords": ["150", "Mbps", "ميجابت"],
        "category": "باقات"
    },
    {
        "question": "ما هي باقة المنصة الترفيهية؟",
        "expected_keywords": ["منصة", "ترفيه", "بث", "أفلام", "مسلسلات"],
        "category": "باقات"
    },
    {
        "question": "ما الفرق بين باقات الفايبر والوايرلس؟",
        "expected_keywords": ["فايبر", "وايرلس", "FTTH", "WiFi"],
        "category": "باقات"
    },
    
    # === أسئلة عن الدفع والتجديد ===
    {
        "question": "كيف يمكنني تجديد باقة الإنترنت؟",
        "expected_keywords": ["تجديد", "وكلاء", "6449", "فروع"],
        "category": "دفع"
    },
    {
        "question": "ما هي طرق الدفع المتاحة؟",
        "expected_keywords": ["نقد", "وكلاء", "بنكي", "تحويل"],
        "category": "دفع"
    },
    
    # === أسئلة عامة ===
    {
        "question": "من هم شركاؤكم؟",
        "expected_keywords": ["تبادل", "اسوار", "تازة", "المنصة"],
        "category": "عام"
    },
    {
        "question": "كم خط كابل ضوئي تديرون؟",
        "expected_keywords": ["70,000", "70000", "سبعين ألف"],
        "category": "معلومات عامة"
    },
    {
        "question": "ما هو المشروع الوطني للإنترنت؟",
        "expected_keywords": ["مشروع وطني", "بنية تحتية", "كابلات"],
        "category": "معلومات عامة"
    }
]


def check_answer_quality(answer: str, expected_keywords: list, must_not_contain: list = None) -> dict:
    """فحص جودة الإجابة"""
    answer_lower = answer.lower()
    
    # فحص الكلمات المفتاحية المتوقعة
    found_keywords = []
    missing_keywords = []
    
    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    # فحص الكلمات الممنوعة
    forbidden_found = []
    if must_not_contain:
        for forbidden in must_not_contain:
            if forbidden.lower() in answer_lower:
                forbidden_found.append(forbidden)
    
    # حساب النتيجة
    keyword_score = len(found_keywords) / len(expected_keywords) if expected_keywords else 1.0
    has_forbidden = len(forbidden_found) > 0
    
    # النتيجة النهائية: يجب أن تحتوي على 70% على الأقل من الكلمات المفتاحية ولا تحتوي على كلمات ممنوعة
    is_correct = keyword_score >= 0.7 and not has_forbidden
    
    return {
        "is_correct": is_correct,
        "keyword_score": keyword_score,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "forbidden_found": forbidden_found,
        "has_forbidden": has_forbidden
    }


def run_tests():
    """تشغيل جميع الاختبارات"""
    print("=" * 80)
    print("🧪 بدء اختبار دقة نموذج RAG")
    print("=" * 80)
    print()
    
    results = []
    total_questions = len(TEST_QUESTIONS)
    correct_answers = 0
    
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        question = test_case["question"]
        expected_keywords = test_case.get("expected_keywords", [])
        must_not_contain = test_case.get("must_not_contain", [])
        category = test_case.get("category", "عام")
        
        print(f"❓ السؤال {i}/{total_questions} [{category}]:")
        print(f"   {question}")
        
        try:
            # الحصول على الإجابة
            answer = get_answer(question)
            
            # فحص جودة الإجابة
            quality = check_answer_quality(answer, expected_keywords, must_not_contain)
            
            # النتيجة
            status = "✅ صحيح" if quality["is_correct"] else "❌ خطأ"
            if quality["is_correct"]:
                correct_answers += 1
            
            print(f"   {status}")
            print(f"   📊 النتيجة: {quality['keyword_score']*100:.1f}% من الكلمات المفتاحية موجودة")
            
            if quality["found_keywords"]:
                print(f"   ✅ كلمات موجودة: {', '.join(quality['found_keywords'][:5])}")
            
            if quality["missing_keywords"]:
                print(f"   ⚠️  كلمات مفقودة: {', '.join(quality['missing_keywords'][:5])}")
            
            if quality["forbidden_found"]:
                print(f"   🚫 كلمات ممنوعة موجودة: {', '.join(quality['forbidden_found'])}")
            
            print(f"   💬 الإجابة: {answer[:150]}...")
            print()
            
            results.append({
                "question": question,
                "category": category,
                "answer": answer,
                "quality": quality,
                "is_correct": quality["is_correct"]
            })
            
        except Exception as e:
            print(f"   ❌ خطأ: {str(e)}")
            print()
            results.append({
                "question": question,
                "category": category,
                "answer": f"خطأ: {str(e)}",
                "quality": {"is_correct": False, "keyword_score": 0},
                "is_correct": False
            })
    
    # حساب الدقة الإجمالية
    accuracy = (correct_answers / total_questions) * 100
    
    # حساب الدقة حسب الفئة
    category_stats = {}
    for result in results:
        cat = result["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if result["is_correct"]:
            category_stats[cat]["correct"] += 1
    
    # طباعة التقرير النهائي
    print("=" * 80)
    print("📊 تقرير النتائج")
    print("=" * 80)
    print()
    print(f"✅ الإجابات الصحيحة: {correct_answers}/{total_questions}")
    print(f"📈 الدقة الإجمالية: {accuracy:.2f}%")
    print()
    
    print("📋 الدقة حسب الفئة:")
    print("-" * 80)
    for cat, stats in sorted(category_stats.items()):
        cat_accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"   {cat:20s}: {stats['correct']:2d}/{stats['total']:2d} ({cat_accuracy:5.1f}%)")
    print()
    
    # الأسئلة الخاطئة
    wrong_answers = [r for r in results if not r["is_correct"]]
    if wrong_answers:
        print("❌ الأسئلة التي فشلت:")
        print("-" * 80)
        for result in wrong_answers:
            print(f"   • [{result['category']}] {result['question']}")
            print(f"     النتيجة: {result['quality']['keyword_score']*100:.1f}%")
            if result['quality']['forbidden_found']:
                print(f"     كلمات ممنوعة: {', '.join(result['quality']['forbidden_found'])}")
        print()
    
    # الأسئلة الصحيحة
    correct_results = [r for r in results if r["is_correct"]]
    if correct_results:
        print(f"✅ الأسئلة الصحيحة ({len(correct_results)}):")
        print("-" * 80)
        for result in correct_results[:10]:  # أول 10 فقط
            print(f"   • [{result['category']}] {result['question']}")
        if len(correct_results) > 10:
            print(f"   ... و {len(correct_results) - 10} أسئلة أخرى")
        print()
    
    # حفظ النتائج في ملف JSON
    output_file = BASE_DIR / "tests" / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "category_stats": {k: {"total": v["total"], "correct": v["correct"], 
                               "accuracy": (v["correct"] / v["total"]) * 100 if v["total"] > 0 else 0} 
                           for k, v in category_stats.items()},
        "results": [
            {
                "question": r["question"],
                "category": r["category"],
                "answer": r["answer"][:500],  # تقليل الطول
                "is_correct": r["is_correct"],
                "keyword_score": r["quality"]["keyword_score"],
                "found_keywords": r["quality"]["found_keywords"],
                "missing_keywords": r["quality"]["missing_keywords"],
                "forbidden_found": r["quality"]["forbidden_found"]
            }
            for r in results
        ]
    }
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"💾 تم حفظ النتائج في: {output_file}")
        print()
    except Exception as e:
        print(f"⚠️  لم يتم حفظ النتائج: {e}")
        print()
    
    return {
        "total": total_questions,
        "correct": correct_answers,
        "accuracy": accuracy,
        "category_stats": category_stats,
        "results": results,
        "output_file": str(output_file)
    }


if __name__ == "__main__":
    print("🚀 بدء تشغيل اختبارات دقة النموذج...")
    print()
    print("⚠️  ملاحظة: تأكد من إعادة بناء قاعدة البيانات قبل الاختبار:")
    print("   python scripts/rebuild_vectorstore.py")
    print()
    
    try:
        results = run_tests()
        
        print("=" * 80)
        print("✅ اكتمل الاختبار!")
        print("=" * 80)
        print()
        print(f"📊 الدقة النهائية: {results['accuracy']:.2f}%")
        print(f"📁 ملف النتائج: {results.get('output_file', 'غير محفوظ')}")
        
    except KeyboardInterrupt:
        print("\n⚠️  تم إيقاف الاختبار بواسطة المستخدم")
    except Exception as e:
        print(f"\n❌ خطأ في تشغيل الاختبار: {e}")
        import traceback
        traceback.print_exc()

