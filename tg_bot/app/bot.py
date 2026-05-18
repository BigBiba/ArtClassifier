import asyncio
import logging

from aiogram import Bot, Dispatcher
from decouple import config

from .handlers import router
from .ml_client import MLClient


logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    token = config("BOT_TOKEN")
    ml_service_url = config("ML_SERVICE_URL", default="http://ml_service:8000")

    bot = Bot(token=token)
    dp = Dispatcher()

    dp.include_router(router)

    async with MLClient(base_url=ml_service_url) as ml_client:
        logger.info("Telegram bot started")

        try:
            await dp.start_polling(
                bot,
                ml_client=ml_client
            )
        finally:
            await bot.session.close()
            logger.info("Telegram bot stopped")


if __name__ == "__main__":
    asyncio.run(main())