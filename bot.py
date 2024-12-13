from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import asyncio
import config
import time

bot = Bot(token=config.TOKEN)
dp = Dispatcher(bot)


async def error(chat_id):
    await bot.send_message(chat_id=chat_id, text='Что-то пошло не так...')


async def edit_message_with_typing(chat_id, message_id, text, chunk_size=5, delay=0.4):
    try:
        words = text.split()
        displayed_text = ""
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            displayed_text += (chunk + " ")
            await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=displayed_text + "●")
            await asyncio.sleep(delay)
        await bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=displayed_text.strip())

    except Exception as er:
        print(f"Ошибка: {er}", flush=True)
        await error(chat_id)


@dp.message_handler(commands=['start'])
async def welcome(message: types.Message):
    try:
        texts = f"Добро пожаловать, {message.from_user.first_name}!\nЧем могу помочь?"
        await bot.send_chat_action(chat_id=message.chat.id, action='typing')

        sent_message = await bot.send_message(chat_id=message.chat.id, text="●")
        await edit_message_with_typing(message.chat.id, sent_message.message_id, texts)

    except Exception as er:
        print(f"Ошибка: {er}", flush=True)
        await error(message.chat.id)


@dp.message_handler(commands=['help'])
async def helps(message: types.Message):
    try:
        text = "Я - виртуальный помощник Т - банка.\nПомогу ответить на любой ваш вопрос!"
        await bot.send_chat_action(chat_id=message.chat.id, action='typing')

        sent_message = await bot.send_message(chat_id=message.chat.id, text="●")
        await edit_message_with_typing(message.chat.id, sent_message.message_id, text)

    except Exception as er:
        print(f"Ошибка: {er}", flush=True)
        await error(message.chat.id)


@dp.message_handler(content_types=['text'])
async def mgs(message: types.Message):
    try:
        await bot.send_chat_action(chat_id=message.chat.id, action='typing')
        texts, urls_parsed = config.result(q=message.text)

        await bot.send_chat_action(chat_id=message.chat.id, action='typing')
        sent_message = await bot.send_message(chat_id=message.chat.id, text="●")
        await edit_message_with_typing(message.chat.id, sent_message.message_id, texts)

        if "нет ответа" in texts.lower():
            return

        urls = "Источники:\n"
        for item in urls_parsed:
            urls += item + '\n'
        await bot.send_message(chat_id=message.chat.id, text=urls, disable_web_page_preview=True)

    except Exception as er:
        print(f"Ошибка: {er}", flush=True)
        await error(message.chat.id)


@dp.message_handler(content_types=['voice', 'video', 'audio', 'document', 'sticker', 'photo', 'location',
                                   'contact', 'poll', 'photo', 'video_note'])
async def other(message: types.Message):
    try:
        time.sleep(0.5)
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)

    except:
        await error(message.chat.id)


if __name__ == '__main__':
    try:
        executor.start_polling(dp, skip_updates=False)

    except Exception as e:
        time.sleep(3)
