import logging

import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from decouple import config

from .handlers import handle_message, handle_group_join, handle_allow_message
from .ml_client import MLClient


logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    vk_token = config("BOT_TOKEN")
    group_id = config("VK_GROUP_ID", cast=int)
    ml_service_url = config("ML_SERVICE_URL", default="http://ml_service:8000")

    vk_session = vk_api.VkApi(token=vk_token)
    vk = vk_session.get_api()

    longpoll = VkBotLongPoll(vk_session, group_id)

    logger.info("VK bot started")

    with MLClient(base_url=ml_service_url) as ml_client:
        for event in longpoll.listen():
            if event.type == VkBotEventType.MESSAGE_NEW:
                try:
                    handle_message(
                        vk=vk,
                        message=event.obj.message,
                        ml_client=ml_client
                    )
                except Exception as e:
                    logger.exception(f"Ошибка при обработке сообщения VK: {e}")
            elif event.type == VkBotEventType.GROUP_JOIN:
                handle_group_join(
                    vk=vk,
                    event_obj=event.obj
                )
            elif event.type == VkBotEventType.MESSAGE_ALLOW:
                handle_allow_message(
                    vk=vk,
                    event_obj=event.obj
                )


if __name__ == "__main__":
    main()