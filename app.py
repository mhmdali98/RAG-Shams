"""
app.py
واجهة دردشة تفاعلية لبوت شركة الشمس تيليكوم باستخدام Gradio
(نسخة محسّنة مستقرة مع دعم كامل لإصدارات Gradio الحديثة)
"""

import logging
import gradio as gr
from rag_engine import get_answer

# إعداد السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def sanitize_input(message: str) -> tuple[bool, str]:
    if not message or not message.strip():
        return False, "مرحباً! 😊 يبدو أنك لم تكتب سؤالك بعد. كيف يمكنني مساعدتك اليوم؟"
    
    clean_msg = message.strip()
    if len(clean_msg) < 2:
        return False, "عذراً، السؤال قصير جداً. يمكنك سؤالي عن خدماتنا، باقاتنا، أو أي معلومات عن شركة الشمس تيليكوم. 😊"
    
    # حماية من محاولات الحقن (بسيطة وآمنة)
    dangerous = ["<script", "javascript:", "onload=", "onerror="]
    if any(d in clean_msg.lower() for d in dangerous):
        return False, "عذراً، لا يمكن معالجة هذا النوع من المحتوى. يرجى كتابة سؤالك بشكل عادي. 😊"
    
    return True, clean_msg


def respond(message: str, chat_history: list) -> tuple[str, list]:
    """
    معالجة الرسالة باستخدام تنسيق الرسائل الجديد:
    chat_history = [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    try:
        if chat_history is None:
            chat_history = []

        # تنقية المدخل
        is_valid, result = sanitize_input(message)
        if not is_valid:
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": result})
            return "", chat_history

        # الحصول على الإجابة
        bot_response = get_answer(result)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})

        return "", chat_history

    except Exception as e:
        logger.error(f"خطأ في معالجة الرسالة: {e}", exc_info=True)
        error_msg = (
            "عذراً، حدث خطأ فني أثناء معالجة سؤالك. 😔\n"
            "يمكنك المحاولة لاحقًا أو التواصل مباشرةً على:\n"
            "📞 6449 | 📧 info@shams-tele.com"
        )
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        return "", chat_history


# واجهة Gradio نظيفة ومستقرة
with gr.Blocks(title="🌞 بوت شركة الشمس تيليكوم") as demo:
    gr.Markdown("## 🌞 مرحباً بك في بوت شركة الشمس تيليكوم")
    gr.Markdown("### اسألني عن الباقات، الأسعار، الخدمات، أو أي معلومة عن شمس تيليكوم! 😊")
    
    gr.Markdown("---")
    
    chatbot = gr.Chatbot(
        height=500,
        label="الدردشة",
        value=[],  # دائماً قائمة فارغة من tuples
    )
    
    msg = gr.Textbox(
        placeholder="💬 مثال: ما سعر باقة فايبر 75؟",
        label="اكتب سؤالك",
        lines=2
    )
    
    with gr.Row():
        submit_btn = gr.Button("📤 إرسال", variant="primary")
        clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ مسح")

    # ربط الأحداث
    msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    submit_btn.click(fn=respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    
    gr.Markdown("---")
    
    with gr.Accordion("💡 أمثلة على الأسئلة", open=False):
        gr.Markdown("""
        - ما هي أسعار باقات الإنترنت؟
        - كم سعر باقة Sun؟
        - هل الدعم الفني متاح 24 ساعة؟
        - ما هي مناطق التغطية؟
        - كيف أتواصل معكم؟
        """)

    with gr.Accordion("📞 معلومات التواصل", open=False):
        gr.Markdown("""
        **الهاتف:** 6449  
        **الواتساب:** [اضغط هنا](https://api.whatsapp.com/send/?phone=9647856669616)  
        **البريد:** info@shams-tele.com  
        **الموقع:** بغداد (شارع الصناعة) | ديالى
        """)

if __name__ == "__main__":
    logger.info("🚀 تشغيل واجهة Gradio...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )