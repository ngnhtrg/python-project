import logging
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from telegram import Message
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

MAX_MESSAGE_LENGTH = 4096
# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(name)


# Define a few command handlers
# These usually take the two arguments update and
# context. Error handlers also receive the raised
# TelegramError object in error
async def start(update, context):
    """Send a message when the command /start is issued."""
    text = """
Welcome to NewsBot
This Bot will help you generate text from prompt
Check /help for more informations
"""
    await update.message.reply_text(text)


async def help(update, context):
    """Send a message when the command /help is issued."""
    text = """
These are what I can do: ...
"""
    await update.message.reply_text(text)


async def gen_img(update, context):
    """Generate image from prompt"""
    prompt = update.message.text
    max_length = context.bot_data.get('max_message_length', 100)
    if len(prompt) > max_length:
        update.message.reply_text(f"Your message is too long (max {max_length} characters).")
    else:
        loading_message: Message = await update.message.reply_text("🔄 Generating image, please wait...")
        image_path = 'generated.png'
        try:
            pipe = context.bot_data.get('default_pipe')
            image = pipe(prompt).images[0]
            image.save(image_path)
            with open(image_path, 'rb') as image:
                await update.message.reply_photo(photo=image, caption=f"Here is your image for: '{prompt}'")
        except Exception as e:
            await update.message.reply_text(f"An error occurred: {e}")


def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler)
        pipe = pipe.to("cpu")

    """Start the bot."""
    api = '7953263769:AAEAOiaJyREUUYraiAKAz6A_LXHgdsr3ZL0'
    app = ApplicationBuilder().token(api).build()

    # Define parameters
    app.bot_data['default_pipe'] = pipe
    app.bot_data['max_message_length'] = 100

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help))
    app.add_handler(MessageHandler(filters=filters.TEXT & ~filters.COMMAND, callback=gen_img))

    # log all errors
    app.add_error_handler(error)

    # Start the Bot
    print("Bot is running...")
    app.run_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    # updater.idle()


if name == 'main':
    main()