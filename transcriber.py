import whisper


def transcribe_video(video_path: str, model_name: str = "small") -> dict:
    """
    Распознаёт речь в видео с помощью Whisper.

    :param video_path: путь к видеофайлу
    :param model_name: название модели Whisper
    :return: результат распознавания с текстом и временными метками
    """
    model = whisper.load_model(model_name)

    result = model.transcribe(
        video_path,
        task="transcribe",
        verbose=False
    )

    return result