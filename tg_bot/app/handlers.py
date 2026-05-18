import logging

from aiogram import Router, F, types
from aiogram.filters import CommandStart

from .ml_client import MLClient


logger = logging.getLogger(__name__)
router = Router()


@router.message(CommandStart())
async def cmd_start(message: types.Message):
    logger.info("Bot started")
    await message.answer(
        "Привет! Пришли мне фото, и я скажу, что на нем изображено."
    )


@router.message(F.photo)
async def handle_photo(message: types.Message, ml_client: MLClient):
    logger.info("Image detected")

    status_msg = await message.reply("Обрабатываю изображение...")

    try:
        photo = message.photo[-1]
        file_info = await message.bot.get_file(photo.file_id)
        photo_file = await message.bot.download_file(file_info.file_path)

        image_bytes = photo_file.read()

        prediction = await ml_client.predict(image_bytes)

        answer = (
            f"Класс: {prediction.class_name}\n"
            f"Подкласс: {prediction.subclass_name}"
        )

        await status_msg.edit_text(
            answer,
            parse_mode="HTML"
        )

    except Exception as e:
        logger.exception(f"Ошибка при обработке изображения: {e}")
        await status_msg.edit_text(
            "Упс, что-то пошло не так при анализе фото."
        )