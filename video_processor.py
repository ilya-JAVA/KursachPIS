import subprocess
from pathlib import Path


def burn_subtitles(
    video_path: str,
    subtitles_path: str,
    output_path: str
) -> str:
    """Встраивает субтитры в видео с помощью FFmpeg."""

    video = Path(video_path).resolve()
    subtitles = Path(subtitles_path).resolve()
    output = Path(output_path).resolve()

    if not video.is_file():
        raise FileNotFoundError(f"Видео не найдено: {video}")

    if not subtitles.is_file():
        raise FileNotFoundError(
            f"Файл субтитров не найден: {subtitles}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    # Передаём FFmpeg путь через параметр filename,
    # чтобы двоеточия и другие символы не ломали фильтр.
    subtitle_filter_path = (
        str(subtitles)
        .replace("\\", "/")
        .replace(":", r"\:")
        .replace("'", r"\'")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace(",", r"\,")
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vf", f"subtitles=filename='{subtitle_filter_path}'",
        "-c:a", "copy",
        str(output)
    ]

    subprocess.run(command, check=True)

    return str(output)
