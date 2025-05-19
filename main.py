#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Telegram bot that scans images in groups for NSFW content and removes them.
Main entry point of the application.
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from handlers.gif import NSFWGifDetector

# Load environment variables from .env file
load_dotenv()

# Add the project root to sys.path to allow importing from handlers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from handlers.photo import NSFWDetector
from handlers.sticker import NSFWStickerDetector

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize NSFW detector
current_dir = os.path.dirname(os.path.abspath(__file__))
nsfw_detector = NSFWDetector(current_dir)
sticker_detector = NSFWStickerDetector(current_dir)
gif_detector = NSFWGifDetector(current_dir)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text('سلام! من یک بات تشخیص و حذف محتوای نامناسب هستم.')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text('من عکس‌ها و استیکرهای ارسالی را بررسی می‌کنم و اگر محتوای نامناسب تشخیص داده شود، آن را حذف می‌کنم.')

async def scan_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan photos for NSFW content and delete if necessary."""
    if update.message and update.message.photo:
        # Get the largest photo available
        photo = update.message.photo[-1]
        
        # Create a temp directory if it doesn't exist
        temp_dir = os.path.join(current_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create a temporary file path
        temp_file = os.path.join(temp_dir, f"{update.message.message_id}.jpg")
        
        try:
            # Download the photo
            file = await context.bot.get_file(photo.file_id)
            await file.download_to_drive(temp_file)
            
            logger.info(f"Downloaded photo to {temp_file}, scanning...")
            
            # Check if the image is NSFW
            is_nsfw, nsfw_probability, details = nsfw_detector.is_nsfw(temp_file)
            
            if is_nsfw:
                # Delete the message
                await context.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=update.message.message_id
                )
                logger.info(f"Deleted NSFW image from chat {update.effective_chat.id}. Details: {details}")
                
                # Notify about deletion (optional)
                user_name = update.message.from_user.first_name
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"⚠️ یک گور نامناسب از کاربر {user_name} پاک شد"
                )
            else:
                logger.info(f"Image is safe. Details: {details}")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

async def scan_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan stickers for NSFW content and delete if necessary."""
    if update.message and update.message.sticker:
        sticker = update.message.sticker
        
        # Create a temp directory if it doesn't exist
        temp_dir = os.path.join(current_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Determine file extension based on sticker type
        if sticker.is_video:
            logger.info("Video sticker detected.")
            ext = ".webm"
        elif sticker.is_animated:
            logger.info("Animated sticker detected.")
            ext = ".tgs"
        else:
            logger.info("Static sticker detected.")
            ext = ".webp"
        
        # Create a temporary file path
        temp_file = os.path.join(temp_dir, f"sticker_{update.message.message_id}{ext}")
        
        try:
            # Download the sticker
            file = await context.bot.get_file(sticker.file_id)
            await file.download_to_drive(temp_file)
            
            logger.info(f"Downloaded sticker to {temp_file}, scanning...")
            
            # Check if the sticker is NSFW
            is_nsfw, nsfw_probability, details = sticker_detector.is_nsfw(temp_file)
            
            if is_nsfw:
                # Delete the message
                await context.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=update.message.message_id
                )
                logger.info(f"Deleted NSFW sticker from chat {update.effective_chat.id}. Details: {details}")
                
                # Notify about deletion (optional)
                user_name = update.message.from_user.first_name
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"⚠️ یک گور نامناسب از کاربر {user_name} پاک شد. "
                )
            else:
                logger.info(f"Sticker is safe. Details: {details}")
        except Exception as e:
            logger.error(f"Error processing sticker: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

async def scan_animation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan animated GIFs and videos for NSFW content and delete if necessary."""
    if update.message and update.message.animation:
        animation = update.message.animation
        
        # Create a temp directory if it doesn't exist
        temp_dir = os.path.join(current_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Get the file extension from the mime type
        mime_type = animation.mime_type
        if mime_type == "video/mp4":
            ext = ".mp4"
        else:
            ext = ".gif"
        
        # Create a temporary file path
        temp_file = os.path.join(temp_dir, f"animation_{update.message.message_id}{ext}")
        
        try:
            # Download the animation
            file = await context.bot.get_file(animation.file_id)
            await file.download_to_drive(temp_file)
            
            logger.info(f"Downloaded animation to {temp_file}, scanning...")
            
            # Check if the animation is NSFW
            is_nsfw, nsfw_probability, details = gif_detector.is_nsfw(temp_file)
            
            if is_nsfw:
                # Delete the message
                await context.bot.delete_message(
                    chat_id=update.effective_chat.id,
                    message_id=update.message.message_id
                )
                logger.info(f"Deleted NSFW animation from chat {update.effective_chat.id}. Details: {details}")
                
                # Notify about deletion (optional)
                user_name = update.message.from_user.first_name
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"⚠️ یک گور نامناسب از کاربر {user_name} پاک شد. "
                )
            else:
                logger.info(f"Animation is safe. Details: {details}")
        except Exception as e:
            logger.error(f"Error processing animation: {str(e)}")
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

def main() -> None:
    """Start the bot."""
    # Create the Application
    token = os.environ.get("TELEGRAM_TOKEN")
    
    # Check if TELEGRAM_TOKEN is set
    if not token:
        print("Error: TELEGRAM_TOKEN environment variable not set!")
        print("Please set your Telegram bot token in a .env file or environment variable")
        sys.exit(1)
        
    print("Starting NSFW Telegram Bot...")
    
    application = Application.builder().token(token).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Add message handler for photos
    application.add_handler(MessageHandler(filters.PHOTO, scan_photo))
    
    # Add message handler for stickers
    application.add_handler(MessageHandler(filters.Sticker.ALL, scan_sticker))
    
    # Add message handler for animations
    application.add_handler(MessageHandler(filters.ANIMATION, scan_animation))

    # Start the Bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    logger.info("Bot started")

if __name__ == '__main__':
    main()
