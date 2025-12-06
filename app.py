"""
app.py
واجهة دردشة تفاعلية محسّنة لبوت شركة الشمس تيليكوم باستخدام Gradio
مع نظام اقتراحات ذكي ومؤشر تحميل
"""

import logging
import gradio as gr
from rag_engine import get_answer, vectorstore
from suggestions import get_suggestions, get_related_suggestions

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


def respond(message: str, chat_history: list) -> tuple[str, list, gr.update]:
    """
    معالجة الرسالة مع توليد اقتراحات تلقائية
    """
    try:
        if chat_history is None:
            chat_history = []

        # تنقية المدخل
        is_valid, result = sanitize_input(message)
        if not is_valid:
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": result})
            # اقتراحات عامة عند خطأ في الإدخال
            suggestions = get_suggestions("", num_suggestions=4)
            return "", chat_history, gr.update(choices=suggestions, value=None)

        # الحصول على الإجابة
        bot_response = get_answer(result)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_response})

        # توليد اقتراحات ذكية بناءً على السؤال والإجابة
        try:
            suggestions = get_related_suggestions(result, vectorstore, num_suggestions=4)
        except:
            suggestions = get_suggestions(result, num_suggestions=4)

        return "", chat_history, gr.update(choices=suggestions, value=None)

    except Exception as e:
        logger.error(f"خطأ في معالجة الرسالة: {e}", exc_info=True)
        error_msg = (
            "عذراً، حدث خطأ فني أثناء معالجة سؤالك. 😔\n"
            "يمكنك المحاولة لاحقًا أو التواصل مباشرةً على:\n"
            "📞 6449 | 📧 info@shams-tele.com"
        )
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": error_msg})
        suggestions = get_suggestions("", num_suggestions=4)
        return "", chat_history, gr.update(choices=suggestions, value=None)


def on_suggestion_select(selected_value: str, chat_history: list) -> tuple[str, list, gr.update]:
    """
    معالجة اختيار اقتراح من القائمة
    """
    if selected_value:
        # استخدام الاقتراح المختار كسؤال
        return respond(selected_value, chat_history)
    return "", chat_history, gr.update()


# واجهة Gradio محسّنة مع اقتراحات
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🌞 مرحباً بك في بوت شركة الشمس تيليكوم
    
    ### اسألني عن الباقات، الأسعار، الخدمات، أو أي معلومة عن شمس تيليكوم! 😊
    
    **بوت ذكي يجيب بدقة بناءً على معلوماتنا الرسمية فقط**
    """)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                label="الدردشة",
                value=[]
            )
            
            msg = gr.Textbox(
                placeholder="💬 مثال: ما سعر باقة فايبر 75؟",
                label="اكتب سؤالك",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("📤 إرسال")
                clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ مسح المحادثة")
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 أسئلة مقترحة")
            suggestions = gr.Radio(
                choices=get_suggestions("", num_suggestions=4),
                label="اختر سؤالاً أو اكتب سؤالك أدناه",
                type="value",
                interactive=True,
                show_label=True
            )
            
            gr.Markdown("---")
            
            with gr.Accordion("📞 معلومات التواصل", open=False):
                gr.Markdown("""
                **الهاتف:** 6449  
                **الواتساب:** [اضغط هنا](https://api.whatsapp.com/send/?phone=9647856669616)  
                **البريد:** info@shams-tele.com  
                **الموقع:** بغداد (شارع الصناعة) | ديالى
                """)

    gr.Markdown("---")
    
    with gr.Accordion("ℹ️ معلومات إضافية", open=False):
        gr.Markdown("""
        ### 💡 نصائح للاستخدام:
        - يمكنك كتابة سؤالك مباشرة أو اختيار أحد الأسئلة المقترحة
        - البوت يجيب باللغة العربية فقط بناءً على معلوماتنا الرسمية
        - بعد كل إجابة، ستظهر أسئلة مقترحة جديدة متعلقة بموضوعك
        
        ### ✨ المميزات:
        - إجابات دقيقة من قاعدة بياناتنا الرسمية
        - اقتراحات ذكية للأسئلة
        - دعم كامل للغة العربية
        - واجهة سهلة الاستخدام
        """)

    # ربط الأحداث
    msg.submit(
        fn=respond, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, suggestions]
    )
    submit_btn.click(
        fn=respond, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, suggestions]
    )
    
    # عند اختيار اقتراح
    suggestions.change(
        fn=on_suggestion_select,
        inputs=[suggestions, chatbot],
        outputs=[msg, chatbot, suggestions]
    )

if __name__ == "__main__":
    logger.info("🚀 تشغيل واجهة Gradio...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )