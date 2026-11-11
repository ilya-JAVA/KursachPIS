from pathlib import Path

from transcriber import transcribe_video
from subtitle_generator import save_srt
from video_processor import burn_subtitles


def process_video(
    video_path: str,
    model_name: str = "small"
) -> tuple[str, str]:
    """Создаёт субтитры и встраивает их в видео."""

    video = Path(video_path)

    if not video.is_file():
        raise FileNotFoundError("Выбранный видеофайл не найден.")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    subtitles_path = output_dir / f"{video.stem}_{model_name}.srt"
    output_video_path = output_dir / f"{video.stem}_subtitled.mp4"

    print("Шаг 1/3: распознаём речь...")
    result = transcribe_video(str(video), model_name=model_name)

    print("Шаг 2/3: сохраняем субтитры...")
    save_srt(result, str(subtitles_path))

    print("Шаг 3/3: встраиваем субтитры в видео...")
    burn_subtitles(
        str(video),
        str(subtitles_path),
        str(output_video_path)
    )

    print("Готово!")
    print(f"Видео: {output_video_path}")
    print(f"Субтитры: {subtitles_path}")

    return str(output_video_path), str(subtitles_path)


def process_audio(audio_path: str, model_name: str = "small") -> str:
    """Создаёт субтитры из отдельного аудиофайла."""

    audio = Path(audio_path)

    if not audio.is_file():
        raise FileNotFoundError("Выбранный аудиофайл не найден.")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    subtitles_path = output_dir / f"{audio.stem}_{model_name}.srt"

    print("Шаг 1/2: распознаём речь...")
    result = transcribe_video(str(audio), model_name=model_name)

    print("Шаг 2/2: сохраняем субтитры...")
    save_srt(result, str(subtitles_path))

    print("Готово!")
    print(f"Субтитры: {subtitles_path}")

    return str(subtitles_path)


if __name__ == "__main__":
    media_path = input("Укажи путь к видео или аудио: ").strip().strip('"')

    try:
        extension = Path(media_path).suffix.lower()

        if extension in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
            process_audio(media_path)
        else:
            process_video(media_path)

    except Exception as error:
        print(f"Ошибка при обработке файла: {error}")
