import logging
import os
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

from photo import NSFWDetector
from gif import NSFWGifDetector
from sticker_static import NSFWStaticStickerDetector
from sticker_animated import NSFWAnimatedStickerDetector

# لاگ‌برداری
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = "7619620211:AAGY8KCvl5wiP0zhamEKYOmAUmnUzNYasB8"

nsfw_detector = NSFWDetector()
nsfw_gif_detector = NSFWGifDetector()
nsfw_static_sticker_detector = NSFWStaticStickerDetector()
nsfw_animated_sticker_detector = NSFWAnimatedStickerDetector()

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if update.message.photo:
            photo = update.message.photo[-1]
            photo_file = await photo.get_file()
            photo_bytes = await photo_file.download_as_bytearray()

            is_nsfw = nsfw_detector.is_nsfw(photo_bytes)
            if is_nsfw:
                await update.message.delete()
                await context.bot.send_message(chat_id=update.effective_chat.id, text="❌گورپاک شد")
                logger.info("NSFW photo deleted.")
    except Exception as e:
        logger.error(f"Error in handle_photo: {e}")

async def handle_gif(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if update.message.animation:
            gif = update.message.animation
            gif_file = await gif.get_file()
            gif_path = "temp.gif"
            await gif_file.download_to_drive(gif_path)

            is_nsfw = nsfw_gif_detector.is_nsfw(gif_path)
            os.remove(gif_path)

            if is_nsfw:
                await update.message.delete()
                await context.bot.send_message(chat_id=update.effective_chat.id, text="❌گورپاک شد")
                logger.info("NSFW gif deleted.")
    except Exception as e:
        logger.error(f"Error in handle_gif: {e}")

async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sticker = update.message.sticker
        if sticker.is_animated or sticker.is_video:
            file = await sticker.get_file()
            temp_path = "temp_sticker.webm"
            await file.download_to_drive(temp_path)

            is_nsfw = nsfw_animated_sticker_detector.is_nsfw(temp_path)
            os.remove(temp_path)
        else:
            file = await sticker.get_file()
            sticker_bytes = await file.download_as_bytearray()
            is_nsfw = nsfw_static_sticker_detector.is_nsfw(sticker_bytes)

        if is_nsfw:
            await update.message.delete()
            await context.bot.send_message(chat_id=update.effective_chat.id, text="❌گورپاک شد")
            logger.info("NSFW sticker deleted.")
    except Exception as e:
        logger.error(f"Error in handle_sticker: {e}")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.ANIMATION, handle_gif))
    application.add_handler(MessageHandler(filters.Sticker.ALL, handle_sticker))

    logger.info("Bot started.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
