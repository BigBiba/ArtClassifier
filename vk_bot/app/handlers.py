import logging
import random
from typing import Any

import httpx


logger = logging.getLogger(__name__)


def send_message(vk, peer_id: int, text: str) -> None:
    vk.messages.send(
        peer_id=peer_id,
        message=text,
        random_id=random.randint(1, 2_000_000_000)
    )

WELCOME_MESSAGE = "Привет! Пришли мне фото, и я скажу, что на нем изображено."


def handle_group_join(vk, event_obj: Any) -> None:
    user_id = event_obj.user_id

    if user_id is None:
        logger.warning(f"Не удалось определить user_id нового подписчика: {event_obj}")
        return

    try:
        send_message(vk, peer_id=int(user_id), text=WELCOME_MESSAGE)
    except Exception as e:
        logger.exception(f"Ошибка при отправке приветственного сообщения VK: {e}")


def get_largest_photo_url(attachments: list) -> str | None:
    for attachment in attachments:
        if attachment.get("type") == "photo":
            photo = attachment.get("photo", {})
            sizes = photo.get("sizes", [])

            if not sizes:
                return None

            largest = max(
                sizes,
                key=lambda size: size.get("width", 0) * size.get("height", 0)
            )

            return largest.get("url")

    return None


def handle_message(vk, message: dict, ml_client) -> None:
    peer_id = message["peer_id"]
    attachments = message.get("attachments", [])

    photo_url = get_largest_photo_url(attachments)

    if photo_url is None:
        send_message(
            vk,
            peer_id,
            "Пришли, пожалуйста, изображение для классификации."
        )
        return

    send_message(vk, peer_id, "Обрабатываю изображение...")

    try:
        image_response = httpx.get(photo_url, timeout=60.0)
        image_response.raise_for_status()

        prediction = ml_client.predict(image_response.content)

        answer = (
            f"Класс: {prediction.class_name}\n"
            f"Подкласс: {prediction.subclass_name}"
        )

        send_message(
            vk,
            peer_id,
            answer
        )

    except Exception as e:
        logger.exception(f"Ошибка при классификации изображения VK: {e}")
        send_message(
            vk,
            peer_id,
            "Упс, что-то пошло не так при анализе фото."
        )