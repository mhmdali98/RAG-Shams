# scrape_shams_selenium.py
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time
import os

# === إعدادات المتصفح (بدون واجهة رسومية لتسريع التنفيذ) ===
chrome_options = Options()
chrome_options.add_argument("--headless")  # تشغيل خفي (بدون فتح نافذة)
chrome_options.add_argument("--no-sandbox")
chrome_platform = "--disable-dev-shm-usage"
chrome_options.add_argument(chrome_platform)
chrome_options.add_argument("--lang=ar")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

# === قائمة الروابط المطلوبة ===
URLS = [
    "https://shams-tele.com/ar",
    "https://shams-tele.com/ar/service",
    "https://shams-tele.com/ar/project",
    "https://shams-tele.com/ar/jobs",
    "https://shams-tele.com/ar/news",
    "https://shams-tele.com/ar/about",
    
    # أضف أي روابط أخرى تجدها في القائمة
]

def scrape_page(url: str, driver) -> str:
    """استخراج النص من صفحة باستخدام Selenium + BeautifulSoup"""
    try:
        print(f"جارٍ تحميل: {url}")
        driver.get(url)
        time.sleep(3)  # انتظر حتى يكتمل تحميل JS

        # استخراج HTML بعد تحميل JS
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # إزالة العناصر غير المفيدة
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "img"]):
            tag.decompose()

        # حاول استهداف المحتوى الرئيسي
        main = soup.find("main") or soup.find("div", class_="content") or soup.find("section")
        if main:
            text = main.get_text(separator="\n", strip=True)
        else:
            text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""

        # تنظيف الأسطر الفارغة
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    except Exception as e:
        print(f"❌ خطأ في {url}: {e}")
        return ""

def main():
    # تشغيل المتصفح
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    full_content = ""

    try:
        for url in URLS:
            content = scrape_page(url, driver)
            if content:
                full_content += f"\n\n=== مصدر: {url} ===\n{content}"
            time.sleep(1)  # تجنب الحظر

        # حفظ الملف
        output_file = "shams-website-extracted.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_content)

        print(f"✅ تم حفظ المحتوى في: {os.path.abspath(output_file)}")

    finally:
        driver.quit()  # إغلاق المتصفح دائمًا

if __name__ == "__main__":
    main()