from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from photo import NSFWDetector
from gif import NSFWGifDetector
import logging

# توکن ربات تلگرام خود را اینجا قرار دهید
TELEGRAM_BOT_TOKEN = "7473433081:AAGhwQ4ptu_5aLlxjAUGvXD8-RZP_WVDuiY"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ایجاد نمونه از کلاس‌های NSFWDetector و NSFWGifDetector
nsfw_detector = NSFWDetector()
nsfw_gif_detector = NSFWGifDetector()

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندل کردن عکس‌های ارسالی در گروه"""
    if update.message.photo:
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        
        is_nsfw = nsfw_detector.is_nsfw(photo_bytes)
        
        if is_nsfw:
            await update.message.delete()
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="تصویر نامناسب حذف شد."
            )

async def handle_gif(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """هندل کردن گیف‌های ارسالی در گروه"""
    if update.message.animation:
        gif = update.message.animation
        gif_file = await gif.get_file()
        gif_bytes = await gif_file.download_as_bytearray()
        
        is_nsfw = nsfw_gif_detector.is_nsfw(gif_bytes)
        
        if is_nsfw:
            await update.message.delete()
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="گیف نامناسب حذف شد."
            )

def main():
    """تابع اصلی برای اجرای ربات"""
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.ANIMATION, handle_gif))
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
