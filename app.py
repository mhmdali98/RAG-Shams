"""
app.py
واجهة دردشة تفاعلية لبوت شركة الشمس تيليكوم باستخدام Gradio
"""

import logging
import gradio as gr
from rag_engine import get_answer

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_input(message: str) -> tuple[bool, str]:
    """
    التحقق من صحة وتنقية مدخلات المستخدم
    
    Returns:
        (is_valid, error_message): إذا كان المدخل صحيحًا وإلا رسالة الخطأ
    """
    if not message or not message.strip():
        return False, "مرحباً! 😊 يبدو أنك لم تكتب سؤالك بعد. كيف يمكنني مساعدتك اليوم؟"
    
    clean_msg = message.strip()
    
    if len(clean_msg) < 2:
        return False, "عذراً، السؤال قصير جداً. يمكنك سؤالي عن خدماتنا، باقاتنا، أو أي معلومات عن شركة الشمس تيليكوم. 😊"
    
    # حماية من محاولات الحقن
    dangerous_chars = ["<", ">", "{", "}", "script", "alert", "javascript:", "eval(", "function("]
    if any(char in clean_msg.lower() for char in dangerous_chars):
        return False, "عذراً، لا يمكن معالجة هذا النوع من المحتوى. يرجى كتابة سؤالك بشكل عادي. 😊"
    
    return True, clean_msg


def respond(message: str, chat_history: list) -> tuple[str, list]:
    """
    معالجة الرسالة وإرجاع الإجابة
    
    Args:
        message: نص السؤال الجديد
        chat_history: قائمة من الرسائل السابقة
        
    Returns:
        (message, chat_history): نص فارغ والسجل المحدث
    """
    try:
        # التأكد من أن chat_history هو قائمة وليس None
        if chat_history is None:
            chat_history = []
        
        # تحويل chat_history إلى التنسيق الصحيح لـ Gradio 6.x
        # التنسيق: list of dicts with 'role' and 'content' keys
        clean_history = []
        for item in chat_history:
            if isinstance(item, dict) and 'role' in item and 'content' in item:
                # التنسيق الصحيح بالفعل
                clean_history.append(item)
            elif isinstance(item, tuple) and len(item) == 2:
                # تحويل من tuple إلى dict format
                user_msg, bot_msg = item
                if user_msg:
                    clean_history.append({"role": "user", "content": str(user_msg)})
                if bot_msg:
                    clean_history.append({"role": "assistant", "content": str(bot_msg)})
            elif isinstance(item, list) and len(item) == 2:
                # تحويل من list إلى dict format
                user_msg, bot_msg = item
                if user_msg:
                    clean_history.append({"role": "user", "content": str(user_msg)})
                if bot_msg:
                    clean_history.append({"role": "assistant", "content": str(bot_msg)})
        
        chat_history = clean_history
        
        # التحقق من صحة المدخل
        is_valid, result = sanitize_input(message)
        if not is_valid:
            # إضافة رسالة الخطأ - التنسيق: dict with role and content
            chat_history.append({"role": "user", "content": str(message)})
            chat_history.append({"role": "assistant", "content": str(result)})
            return "", chat_history
        
        # الحصول على الإجابة من البوت
        bot_response = get_answer(result)
        
        # إضافة السؤال والإجابة إلى سجل المحادثة
        # التنسيق: dict with 'role' and 'content' keys
        chat_history.append({"role": "user", "content": str(message)})
        chat_history.append({"role": "assistant", "content": str(bot_response)})
        
        return "", chat_history  # نعيد نص فارغ لمسح مربع الإدخال
    
    except Exception as e:
        logger.error(f"خطأ في معالجة الرسالة: {str(e)}")
        logger.exception(e)  # طباعة traceback كامل
        error_msg = "عذراً، حدث خطأ غير متوقع أثناء معالجة سؤالك. 😔\n\nيمكنك:\n• المحاولة مرة أخرى\n• التواصل معنا مباشرة على الرقم 6449\n• إرسال بريد إلكتروني إلى info@shams-tele.com\n\nنعتذر عن الإزعاج ونحن هنا لمساعدتك دائماً! 🌞"
        if chat_history is None:
            chat_history = []
        chat_history.append({"role": "user", "content": str(message)})
        chat_history.append({"role": "assistant", "content": str(error_msg)})
        return "", chat_history

# إنشاء واجهة Gradio احترافية
with gr.Blocks(title="🌞 بوت شركة الشمس تيليكوم") as demo:
    # العنوان الرئيسي
    with gr.Row():
        gr.Markdown("""
        # 🌞 مرحباً بك في بوت شركة الشمس تيليكوم
        
        ### أنا مساعدك الذكي هنا للإجابة على جميع استفساراتك! 
        
        يمكنك سؤالي عن خدماتنا، باقاتنا، أسعارنا، شراكاتنا، أو أي معلومات عن الشركة. 😊
        """)
    
    gr.Markdown("---")
    
    # منطقة المحادثة
    chatbot = gr.Chatbot(
        height=500,
        label="المحادثة",
        value=[]  # تهيئة بقائمة فارغة
    )
    
    # مربع الإدخال
    msg = gr.Textbox(
        placeholder="💬 اكتب سؤالك هنا... مثال: ما هي أسعار باقات الإنترنت؟",
        label="سؤالك",
        lines=2,
        max_lines=5
    )
    
    # أزرار التحكم
    with gr.Row():
        submit_btn = gr.Button("📤 إرسال", variant="primary", scale=3)
        clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ مسح المحادثة", scale=1)

    # ربط الأحداث
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    
    gr.Markdown("---")
    
    # أمثلة على الأسئلة
    with gr.Accordion("💡 أمثلة على الأسئلة الشائعة", open=False):
        gr.Markdown("""
        **عن الشركة:**
        - من نحن؟
        - ما هي خدماتكم؟
        - ما هي قيمكم ومهمتكم؟
        
        **عن الباقات والأسعار:**
        - ما هي أسعار باقات الإنترنت؟
        - ما هي الباقات المتوفرة؟
        - كم سعر باقة فايبر 50؟
        - ما هي باقات WiFi المتوفرة؟
        
        **عن الخدمات:**
        - ما هي خدمات الكابل الضوئي؟
        - ما هي المنصة؟
        - كيف أتواصل معكم؟
        """)
    
    # معلومات التواصل
    with gr.Accordion("📞 معلومات التواصل", open=False):
        gr.Markdown("""
        **رقم الهاتف:** 6449
        
        **البريد الإلكتروني:** info@shams-tele.com
        
        **الموقع:** 
        - الفرع الأول: شارع الصناعة، قرب فلكة الجامعة التكنولوجية، بغداد
        - الفرع الثاني: ديالى
        
        **واتساب:** [اضغط هنا للتواصل](https://api.whatsapp.com/send/?phone=9647856669616)
        """)

# تشغيل التطبيق
if __name__ == "__main__":
    logger.info("🚀 بدء تشغيل تطبيق Gradio...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # غيّره إلى True لمشاركة رابط مؤقت
        show_error=True
    )