WELCOME_MESSAGE = "Привет! Пришли мне фото, и я скажу, что на нем изображено."
IMAGE_PROCESSING_MESSAGE = "Обрабатываю изображение..."
WARNING_MESSAGE = "Упс, что-то пошло не так при анализе фото."
IMAGE_REQUEST_MESSAGE = "Пришли, пожалуйста, изображение для классификации."

def build_prediction_message(
    class_name: str,
    subclass_name: str,
) -> str:
    return (
        "Результат классификации:\n"
        f"Класс: {class_name}\n"
        f"Подкласс: {subclass_name}"
    )