from pathlib import Path

from transcriber import transcribe_video
from subtitle_generator import save_srt
from video_processor import burn_subtitles
from translator import translate_segments
from censor import censor_segments, censor_audio


def process_video(
    video_path: str,
    model_name: str = "small",
    target_language: str = "original",
    censor_profanity: bool = False
) -> tuple[str, str]:
    """Создаёт субтитры, переводит их и встраивает в видео."""

    video = Path(video_path)

    if not video.is_file():
        raise FileNotFoundError("Выбранный видеофайл не найден.")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    subtitles_path = (
        output_dir /
        f"{video.stem}_{model_name}_{target_language}.srt"
    )

    output_video_path = (
        output_dir /
        f"{video.stem}_{target_language}_subtitled.mp4"
    )

    censored_audio_path = (
        output_dir /
        f"{video.stem}_{model_name}_{target_language}_censored.wav"
    )

    print("Шаг 1/5: распознаём речь...")
    result = transcribe_video(str(video), model_name=model_name)

    beep_intervals = []

    if censor_profanity:
        print("Шаг 2/5: цензурируем ненормативную лексику...")

        result["segments"], beep_intervals = censor_segments(
            result["segments"],
            result["language"]
        )

        print("Создаём обработанную аудиодорожку...")

        censor_audio(
            audio_path=str(video),
            beep_intervals=beep_intervals,
            output_path=str(censored_audio_path)
        )
    else:
        print("Шаг 2/5: цензурирование отключено.")

    print("Шаг 3/5: переводим субтитры...")
    result["segments"] = translate_segments(
        segments=result["segments"],
        source_language=result["language"],
        target_language=target_language
    )

    print("Шаг 4/5: сохраняем субтитры...")
    save_srt(result, str(subtitles_path))

    print("Шаг 5/5: встраиваем субтитры в видео...")

    burn_subtitles(
        str(video),
        str(subtitles_path),
        str(output_video_path),
        audio_path=(
            str(censored_audio_path)
            if censor_profanity
            else None
        )
    )

    print("Готово!")
    print(f"Видео: {output_video_path}")
    print(f"Субтитры: {subtitles_path}")

    if censor_profanity:
        print(f"Обработанное аудио: {censored_audio_path}")
        print(f"Количество сигналов: {len(beep_intervals)}")

    return str(output_video_path), str(subtitles_path)


def process_audio(
    audio_path: str,
    model_name: str = "small",
    target_language: str = "original",
    censor_profanity: bool = False
) -> tuple[str, str | None]:
    """Создаёт субтитры из аудио с возможностью перевода и цензурирования."""

    audio = Path(audio_path)

    if not audio.is_file():
        raise FileNotFoundError("Выбранный аудиофайл не найден.")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    subtitles_path = (
        output_dir /
        f"{audio.stem}_{model_name}_{target_language}.srt"
    )

    censored_audio_path = (
        output_dir /
        f"{audio.stem}_{model_name}_{target_language}_censored.wav"
    )

    print("Шаг 1/4: распознаём речь...")
    result = transcribe_video(str(audio), model_name=model_name)

    censored_audio_result = None

    if censor_profanity:
        print("Шаг 2/4: цензурируем ненормативную лексику...")

        result["segments"], beep_intervals = censor_segments(
            result["segments"],
            result["language"]
        )

        censor_audio(
            audio_path=str(audio),
            beep_intervals=beep_intervals,
            output_path=str(censored_audio_path)
        )

        censored_audio_result = str(censored_audio_path)

        print(f"Количество сигналов: {len(beep_intervals)}")
    else:
        print("Шаг 2/4: цензурирование отключено.")

    print("Шаг 3/4: переводим субтитры...")
    result["segments"] = translate_segments(
        segments=result["segments"],
        source_language=result["language"],
        target_language=target_language
    )

    print("Шаг 4/4: сохраняем субтитры...")
    save_srt(result, str(subtitles_path))

    print("Готово!")
    print(f"Субтитры: {subtitles_path}")

    if censored_audio_result:
        print(f"Обработанное аудио: {censored_audio_result}")

    return str(subtitles_path), censored_audio_result


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
