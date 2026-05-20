import logging

from aiogram import Router, F, types
from aiogram.filters import CommandStart

from .ml_client import MLClient
from shared.messages import (
    WELCOME_MESSAGE,
    IMAGE_PROCESSING_MESSAGE,
    WARNING_MESSAGE,
    build_prediction_message
)


logger = logging.getLogger(__name__)
router = Router()


@router.message(CommandStart())
async def cmd_start(message: types.Message):
    logger.info("Bot started")
    await message.answer(
        WELCOME_MESSAGE
    )


@router.message(F.photo)
async def handle_photo(message: types.Message, ml_client: MLClient):
    logger.info("Image detected")

    status_msg = await message.reply(IMAGE_PROCESSING_MESSAGE)

    try:
        photo = message.photo[-1]
        file_info = await message.bot.get_file(photo.file_id)
        photo_file = await message.bot.download_file(file_info.file_path)

        image_bytes = photo_file.read()

        prediction = await ml_client.predict(image_bytes)

        answer = build_prediction_message(
            prediction.class_name,
            prediction.subclass_name
        )

        await status_msg.edit_text(
            answer,
            parse_mode="HTML"
        )

    except Exception as e:
        logger.exception(f"Ошибка при обработке изображения: {e}")
        await status_msg.edit_text(
            WARNING_MESSAGE
        )