import subprocess
from pathlib import Path


def get_video_duration(video_path: Path) -> float:
    """Возвращает длительность видеодорожки в секундах."""

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True
    )

    duration = result.stdout.strip()

    if not duration:
        raise ValueError(
            "Не удалось определить длительность видео."
        )

    return float(duration)


def burn_subtitles(
    video_path: str,
    subtitles_path: str,
    output_path: str,
    audio_path: str | None = None
) -> str:
    """Встраивает субтитры в видео с помощью FFmpeg.
    При необходимости заменяет исходную аудиодорожку обработанной.
    """

    video = Path(video_path).resolve()
    subtitles = Path(subtitles_path).resolve()
    output = Path(output_path).resolve()

    if not video.is_file():
        raise FileNotFoundError(
            f"Видео не найдено: {video}"
        )

    if not subtitles.is_file():
        raise FileNotFoundError(
            f"Файл субтитров не найден: {subtitles}"
        )

    video_duration = get_video_duration(video)

    output.parent.mkdir(parents=True, exist_ok=True)

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
    ]

    if audio_path is not None:
        audio = Path(audio_path).resolve()

        if not audio.is_file():
            raise FileNotFoundError(
                f"Обработанное аудио не найдено: {audio}"
            )

        command.extend(["-i", str(audio)])

        command.extend([
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-vf", f"subtitles=filename='{subtitle_filter_path}'",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-af", "apad",
            "-t", str(video_duration),
        ])

    else:
        command.extend([
            "-vf", f"subtitles=filename='{subtitle_filter_path}'",
            "-c:a", "copy",
        ])

    command.append(str(output))

    subprocess.run(command, check=True)

    return str(output)
