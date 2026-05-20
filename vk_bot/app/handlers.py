import logging
import random
from typing import Any
from shared.messages import (
    WELCOME_MESSAGE,
    IMAGE_PROCESSING_MESSAGE,
    WARNING_MESSAGE,
    IMAGE_REQUEST_MESSAGE,
    build_prediction_message
)

import httpx


logger = logging.getLogger(__name__)

def send_message(vk, peer_id: int, text: str) -> None:
    vk.messages.send(
        peer_id=peer_id,
        message=text,
        random_id=random.randint(1, 2_000_000_000)
    )

def send_welcome_message(vk, event_obj: Any) -> None:
    user_id = event_obj.user_id

    if user_id is None:
        logger.warning(f"Не удалось определить user_id нового подписчика: {event_obj}")
        return

    try:
        send_message(vk, peer_id=int(user_id), text=WELCOME_MESSAGE)
    except Exception as e:
        logger.exception(f"Ошибка при отправке приветственного сообщения VK: {e}")


def handle_group_join(vk, event_obj: Any) -> None:
    send_welcome_message(vk, event_obj)

def handle_allow_message(vk, event_obj: Any) -> None:
    send_welcome_message(vk, event_obj)

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
            IMAGE_REQUEST_MESSAGE
        )
        return

    send_message(vk, peer_id, IMAGE_PROCESSING_MESSAGE)

    try:
        image_response = httpx.get(photo_url, timeout=60.0)
        image_response.raise_for_status()

        prediction = ml_client.predict(image_response.content)

        answer = build_prediction_message(
            prediction.class_name,
            prediction.subclass_name
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
            WARNING_MESSAGE
        )