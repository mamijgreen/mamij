import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from photo import NSFWDetector as PhotoNSFWDetector
from gif import NSFWDetector as GifNSFWDetector
from sticker import NSFWDetector as StickerNSFWDetector

# تنظیم لاگ برای چاپ در ترمینال
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO  # یا logging.DEBUG اگر لاگ‌های بیشتری می‌خوای
)
logger = logging.getLogger(__name__)

# توکن ربات تلگرام خود را اینجا قرار دهید
TELEGRAM_BOT_TOKEN = "7473433081:AAGhwQ4ptu_5aLlxjAUGvXD8-RZP_WVDuiY"

# نمونه‌ها
photo_detector = PhotoNSFWDetector()
gif_detector = GifNSFWDetector()
sticker_detector = StickerNSFWDetector()

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo:
        photo = update.message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        logger.info(f"دریافت عکس از کاربر {update.effective_user.id} در چت {update.effective_chat.id}")
        
        is_nsfw = photo_detector.is_nsfw(photo_bytes)
        logger.info(f"نتیجه بررسی عکس: {'نامناسب' if is_nsfw else 'مناسب'}")
        
        if is_nsfw:
            await update.message.delete()
            logger.info("عکس حذف شد به خاطر نامناسب بودن.")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="تصویر نامناسب حذف شد."
            )

async def handle_gif(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.animation:
        gif_file = await update.message.animation.get_file()
        gif_bytes = await gif_file.download_as_bytearray()
        logger.info(f"دریافت گیف از کاربر {update.effective_user.id} در چت {update.effective_chat.id}")
        
        is_nsfw = gif_detector.is_nsfw(gif_bytes)
        logger.info(f"نتیجه بررسی گیف: {'نامناسب' if is_nsfw else 'مناسب'}")
        
        if is_nsfw:
            await update.message.delete()
            logger.info("گیف حذف شد به خاطر نامناسب بودن.")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="گیف نامناسب حذف شد."
            )

async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.sticker:
        sticker_file = await update.message.sticker.get_file()
        sticker_bytes = await sticker_file.download_as_bytearray()
        logger.info(f"دریافت استیکر از کاربر {update.effective_user.id} در چت {update.effective_chat.id}")

        is_nsfw = sticker_detector.is_nsfw(sticker_bytes)
        logger.info(f"نتیجه بررسی استیکر: {'نامناسب' if is_nsfw else 'مناسب'}")

        if is_nsfw:
            await update.message.delete()
            logger.info("استیکر حذف شد به خاطر نامناسب بودن.")
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="استیکر نامناسب حذف شد."
            )

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.ANIMATION, handle_gif))
    application.add_handler(MessageHandler(filters.STICKER, handle_sticker))

    logger.info("ربات شروع به کار کرد.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
