from telegram.ext import Application, CommandHandler, MessageHandler, filters

from logger import logger
from telegram_handlers import start, handle_message, unknown


async def error_handler(update, context):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

async def on_shutdown(application):
    logger.info("Бот остановлен (выключен)")

async def main_async():
    try:
        logger.info("Бот запускается...")
        app = Application.builder().token(TELEGRAM_TOKEN).build()
    
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        app.add_handler(MessageHandler(filters.COMMAND, unknown))
    
        app.add_error_handler(error_handler)
    
        logger.info("Бот запущен!")
        await app.run_polling()
    except Exception as e:
        logger.exception(f"Ошибка при запуске main_async: {e}")
    finally:
        logger.info("Бот остановлен (выключен)")     
