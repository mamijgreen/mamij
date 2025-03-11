import random
import json
import os
import re
import requests
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import pipeline

# API Key های لازم
GOOGLE_API_KEY = 'AIzaSyAjdQuSCp232VwQCsbvoPfccx0bsS95UDE'
CUSTOM_SEARCH_ENGINE_ID = "e71bc15de1e78480f"
BOT_USERNAME = "@hamalNS_bot"

# مسیر فایل برای ذخیره داده‌ها
MEMORY_FILE = "bot_memory.json"
USER_DATA_FILE = "user_data.json"
SEARCH_CACHE_FILE = "search_cache.json"

# پاسخ‌های پیش‌فرض با لحن سرد و خاکی
RESPONSES = {
    "سلام": ["علیک", "چه خبر", "هستیم", "چطوری", "چخبرا", "بنال", "دم شما گرم، چخبر", "به مولا که اومدی"],
    "خوبی": ["بد نیستم", "میگذره", "چاکریم", "هی، بگی نگی", "سر پام هنوز", "چی بگم واقعاً", "ما اینیم دیگه"],
    "خداحافظ": ["برو به سلامت", "چاکریم", "یاعلی", "فعلاً", "گوربای", "بری برنگردی", "مخلصیم"],
    # سایر پاسخ‌ها...
}

# تحلیل احساسات با استفاده از مدل Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis")

# لود کردن داده‌های حافظه
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    return {}

# لود کردن داده‌های کاربر
def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    return {}

# لود کردن کش جستجو
def load_search_cache():
    if os.path.exists(SEARCH_CACHE_FILE):
        with open(SEARCH_CACHE_FILE, 'r', encoding='utf-8') as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {}
    return {}

# ذخیره داده‌های حافظه
def save_memory(memory_data):
    with open(MEMORY_FILE, 'w', encoding='utf-8') as file:
        json.dump(memory_data, file, ensure_ascii=False, indent=4)

# ذخیره داده‌های کاربر
def save_user_data(user_data):
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as file:
        json.dump(user_data, file, ensure_ascii=False, indent=4)

# ذخیره کش جستجو
def save_search_cache(cache_data):
    with open(SEARCH_CACHE_FILE, 'w', encoding='utf-8') as file:
        json.dump(cache_data, file, ensure_ascii=False, indent=4)

# جستجو در Google Custom Search API
def search_info(query):
    # بررسی کش برای کاهش درخواست‌های تکراری
    cache = load_search_cache()
    if query in cache:
        return cache[query]
    
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={CUSTOM_SEARCH_ENGINE_ID}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        try:
            search_result = data['items'][0]['snippet']
            cache[query] = search_result
            save_search_cache(cache)
            return search_result
        except (IndexError, KeyError):
            return "نتیجه‌ای پیدا نشد"
    else:
        return "خطا در جستجو"

# تحلیل احساسات
def analyze_sentiment(text):
    sentiment = sentiment_analyzer(text)
    return sentiment[0]['label'], sentiment[0]['score']

# یادگیری پاسخ جدید با توانایی جستجو
def learn_response(text, search=True):
    memory = load_memory()
    
    # استخراج کلمات کلیدی
    keywords = re.findall(r'\b\w+\b', text)
    if not keywords:
        keyword = "عمومی"
    else:
        keyword = keywords[0]
    
    # جستجوی اطلاعات آنلاین اگر فعال باشد
    search_result = None
    if search and len(text) > 5:
        search_result = search_info(text)
    
    if keyword not in memory:
        memory[keyword] = {"responses": [], "search_info": []}
    
    is_new = False
    if text not in memory[keyword]["responses"]:
        memory[keyword]["responses"].append(text)
        is_new = True
    
    if search_result and search_result not in memory[keyword]["search_info"]:
        memory[keyword]["search_info"].append(search_result)
        is_new = True
    
    if is_new:
        save_memory(memory)
        return True, search_result
    
    return False, search_result

# ذخیره تعاملات کاربر
async def save_user_interaction(update: Update):
    user_data = load_user_data()
    user_id = str(update.message.from_user.id)
    
    if user_id not in user_data:
        user_data[user_id] = {
            "name": update.message.from_user.first_name,
            "username": update.message.from_user.username,
            "first_seen": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": []
        }
    
    # ثبت پیام
    user_data[user_id]["messages"].append({
        "text": update.message.text,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # به روز رسانی آخرین بازدید
    user_data[user_id]["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    save_user_data(user_data)

# دستور شروع
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "چطوری داش؟\n"
        "من یه ربات ساده‌م. هر چی بگی یاد میگیرم.\n"
        "اگه سوالی داری بگو، شاید جوابی داشته باشم.\n"
        "برای دیدن قابلیت‌ها /help بزن."
    )
    await save_user_interaction(update)

# دستور راهنما
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ببین داداش اینا امکانات منه:\n\n"
        "- یه چیزی بگو جوابتو میدم\n"
        "- با /learn میتونی چیز جدید یادم بدی\n"
        "- با /search میتونم برات دنبال اطلاعات بگردم\n"
        "- با /stats آمار حرف زدنمون رو میبینی\n\n"
        "تو گروه‌ها هم تگم کن میام"
    )
    await save_user_interaction(update)

# دستور یادگیری
async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.replace("/learn", "").strip()
    
    if not text:
        await update.message.reply_text("داداش بعد از /learn یه چیزی بنویس که یاد بگیرم")
        return
    
    success, search_result = learn_response(text)
    
    if success:
        if search_result:
            await update.message.reply_text(f"گرفتم چی میگی. این‌جا هم یه چیزی پیدا کردم:\n{search_result}")
        else:
            await update.message.reply_text("اوکی، گرفتم")
    else:
        await update.message.reply_text("اینو قبلاً شنیدم. یه چیز جدید بگو")
    
    await save_user_interaction(update)

# دستور جستجو
async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.replace("/search", "").strip()
    
    if not query:
        await update.message.reply_text("داداش بعد از /search یه چیزی بنویس که دنبالش بگردم")
        return
    
    search_result = search_info(query)
    await update.message.reply_text(f"این چیزیه که پیدا کردم:\n{search_result}")
    await save_user_interaction(update)

# دستور آمار
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_data = load_user_data()
    user_id = str(update.message.from_user.id)
    
    if user_id in user_data:
        message_count = len(user_data[user_id]["messages"])
        first_seen = user_data[user_id]["first_seen"]
        
        await update.message.reply_text(
            f"آمار ما:\n\n"
            f"- پیام‌های تو: {message_count}\n"
            f"- از تاریخ: {first_seen}\n"
            f"- کلمات یاد گرفته: {len(load_memory())}\n\n"
            f"همینه که هست"
        )
    else:
        await update.message.reply_text("هنوز آماری ندارم ازت")
    
    await save_user_interaction(update)

# آیینه‌وار سازی متن
def mirror_response(text):
    responses = [
        f"خودت میگی {text}",
        f"حالا که گفتی {text}، خب که چی؟",
        f"{text}؟ جدی؟",
        f"چرا میگی {text}؟",
        f"اینکه گفتی {text} خب باشه"
    ]
    return random.choice(responses)

# پاسخ هوشمند
async def smart_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # ثبت اطلاعات تعامل
    await save_user_interaction(update)
    
    # تشخیص نوع چت (گروه یا خصوصی)
    chat_type = update.message.chat.type
    text = update.message.text.lower()
    
    # حذف منشن در گروه‌ها (در صورتی که ربات منشن شود)
    if chat_type in ["group", "supergroup"]:
        if BOT_USERNAME.lower().replace("@", "") not in text.lower():
            return  # اگر ربات منشن نشده در گروه، پاسخ نده
        text = text.replace("@" + BOT_USERNAME.lower().replace("@", ""), "").strip()
    
    # بررسی پاسخ‌های از پیش تعیین شده
    for keyword in RESPONSES:
        if keyword in text:
            await update.message.reply_text(random.choice(RESPONSES[keyword]))
            return
    
    # بررسی پاسخ‌های یادگرفته شده
    memory = load_memory()
    matching_info = []
    potential_responses = []
    matching_keywords = []
    
    # اضافه کردن پاسخ‌های مرتبط
    for keyword in memory:
        if keyword in text:
            matching_keywords.append(keyword)
            
            if memory[keyword]["responses"]:
                potential_responses.extend(memory[keyword]["responses"])
            
            if memory[keyword]["search_info"]:
                matching_info.extend(memory[keyword]["search_info"])
    
    # اگر کلمات کلیدی تطابق داشتند، اما پاسخی پیدا نشد
    # بررسی کل حافظه برای یافتن پاسخ‌هایی که شامل متن ورودی هستند
    if not potential_responses:
        for keyword in memory:
            for response in memory[keyword]["responses"]:
                if text in response.lower():
                    potential_responses.append(response)
    
    # یک لایه دیگر برای یافتن پاسخ‌های مرتبط
    # بررسی کلمات موجود در متن ورودی
    words = re.findall(r'\b\w+\b', text)
    if not potential_responses:
        for word in words:
            if word in memory and len(word) > 2:  # کلمات با طول بیشتر از 2 را بررسی کن
                if memory[word]["responses"]:
                    potential_responses.extend(memory[word]["responses"])
    
    response_types = [
        # 0: پاسخ آیینه‌وار
        lambda: mirror_response(text),
        
        # 1: اطلاعات جستجو‌شده
        lambda: f"یه چیزی شنیدم: {random.choice(matching_info)}" if matching_info else None,
        
        # 2: پاسخ از حافظه
        lambda: random.choice(potential_responses) if potential_responses else None,
        
        # 3: جستجوی جدید
        lambda: f"صبر کن ببینم... {search_info(text)}"
    ]
    
    # اگر پاسخ‌های بالقوه داریم، اولویت بیشتری به آنها بدهیم
    if potential_responses:
        weights = [0.1, 0.2, 0.6, 0.1]  # اولویت بالاتر برای پاسخ‌های موجود در حافظه
    else:
        weights = [0.3, 0.2, 0.1, 0.4]  # اولویت بالاتر برای جستجو و پاسخ آیینه‌وار
    
    # انتخاب یک نوع پاسخ با وزن‌های مشخص شده
    response_type = random.choices(range(len(response_types)), weights=weights, k=1)[0]
    
    # سعی در استفاده از نوع پاسخ انتخاب شده
    response = response_types[response_type]()
    
    # اگر نوع پاسخ انتخاب شده موجود نیست، یکی دیگر را امتحان کن
    if not response:
        # اولویت بندی مجدد پاسخ‌ها
        for i in range(len(response_types)):
            if i != response_type:  # نوع پاسخ قبلی را دوباره امتحان نکن
                fallback_response = response_types[i]()
                if fallback_response:
                    response = fallback_response
                    break
        
        # اگر هیچ پاسخی پیدا نشد، از پاسخ‌های پیش‌فرض استفاده کن
        if not response:
            fallback_responses = [
                "حوصله نداریما",
                "یه چیز دیگه بگو",
                "نفهمیدم چی گفتی",
                "چی میگی داش؟",
                "خب؟ ادامه بده",
                "یجوری حرف بزن بفهمم"
            ]
            response = random.choice(fallback_responses)
    
    # یادگیری خودکار
    learn_response(text, search=False)
    
    # برای اهداف دیباگ اگر مایل هستید
    # print(f"متن: {text}, کلمات کلیدی یافت شده: {matching_keywords}, تعداد پاسخ‌های یافت شده: {len(potential_responses)}")
    
    await update.message.reply_text(response)
    
    # انتخاب یک نوع پاسخ به صورت تصادفی با وزن‌های خاص
    weights = [0.2, 0.3, 0.3, 0.2]  # احتمال هر نوع پاسخ
    response_type = random.choices(range(len(response_types)), weights=weights, k=1)[0]
    
    # سعی در استفاده از نوع پاسخ انتخاب شده
    response = response_types[response_type]()
    
    # اگر نوع پاسخ انتخاب شده موجود نیست، یکی دیگر را امتحان کن
    if not response:
        fallback_responses = [
            "حوصله نداریما",
            "یه چیز دیگه بگو",
            "نفهمیدم چی گفتی",
            "چی میگی داش؟",
            "خب؟ ادامه بده",
            "یجوری حرف بزن بفهمم"
        ]
        response = random.choice(fallback_responses)
    
    # یادگیری خودکار
    learn_response(text, search=False)
    
    await update.message.reply_text(response)

if __name__ == "__main__":
    if not os.path.exists(MEMORY_FILE):
        save_memory({})
    if not os.path.exists(USER_DATA_FILE):
        save_user_data({})
    if not os.path.exists(SEARCH_CACHE_FILE):
        save_search_cache({})
    
    app = Application.builder().token("7473433081:AAH1ISzqMI1l6t6n_H9_JYvVfpvXj-Lzvik").build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("learn", learn_command))
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), smart_response))
    
    print("ربات در حال اجراست...")
    app.run_polling()

