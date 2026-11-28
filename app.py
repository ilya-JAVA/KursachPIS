import json
import shutil
import uuid
import zipfile

from pathlib import Path

import gradio as gr
import requests

from api import parse_timestamp

API_URL = "http://127.0.0.1:8000/transcribe"

RESULTS_DIR = Path("output/gradio_results")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def generate_subtitles(
    file_path,
    model_name,
    target_language,
    censor_profanity,
    subtitle_mode,
    intervals
):
    """Отправляет аудио или видео в FastAPI и получает результаты."""

    if not file_path:
        raise gr.Error("Сначала выберите аудио- или видеофайл.")

    input_file = Path(file_path)

    if not input_file.is_file():
        raise gr.Error("Выбранный файл не найден.")

    extension = input_file.suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise gr.Error(
            "Неподдерживаемый формат файла. "
            "Выберите аудио или видео."
        )

    if subtitle_mode == "multiple":
        parsed_intervals = []

        for interval in intervals:
            try:
                start = parse_timestamp(interval["start"])
                end = parse_timestamp(interval["end"])
            except (ValueError, TypeError, KeyError):
                raise gr.Error(
                    "Некорректное время. Используйте формат HH:MM:SS."
                )

            if start >= end:
                raise gr.Error(
                    "Начало интервала должно быть раньше окончания."
                )

            parsed_intervals.append((start, end))

        for i in range(len(parsed_intervals)):
            for j in range(i + 1, len(parsed_intervals)):
                start1, end1 = parsed_intervals[i]
                start2, end2 = parsed_intervals[j]

                if start1 < end2 and start2 < end1:
                    raise gr.Error(
                        "Вы указали одинаковые или пересекающиеся "
                        "диапазоны времени. Измените интервалы."
                    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_dir = RESULTS_DIR / str(uuid.uuid4())
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        with input_file.open("rb") as uploaded_file:
            response = requests.post(
                API_URL,
                files={
                    "file": (
                        input_file.name,
                        uploaded_file,
                        "application/octet-stream"
                    )
                },
                data={
                    "model_name": model_name,
                    "target_language": target_language,
                    "censor_profanity": str(censor_profanity).lower(),
                    "subtitle_mode": subtitle_mode,
                    "intervals": json.dumps(intervals, ensure_ascii=False)
                },
                timeout=1800
            )

        response.raise_for_status()

        archive_path = result_dir / "results.zip"
        archive_path.write_bytes(response.content)

        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                filename = Path(member.filename).name
                file_extension = Path(filename).suffix.lower()

                if file_extension not in {".mp4", ".srt"}:
                    continue

                output_path = result_dir / filename

                with archive.open(member) as source:
                    with output_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination)

        output_video = next(
            result_dir.glob("*_subtitled.mp4"),
            None
        )

        output_srt = next(
            result_dir.glob("*.srt"),
            None
        )


        if output_srt is None:
            raise gr.Error("Сервер не вернул файл субтитров.")

        if output_video is not None:
            status_message = "Видео с субтитрами и файл SRT успешно созданы!"
        else:
            status_message = "Файл субтитров SRT успешно создан!"

        return (
            str(output_video) if output_video else None,
            str(output_srt),
            status_message
        )

    except gr.Error:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise

    except requests.exceptions.ConnectionError:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error(
            "Не удалось подключиться к FastAPI. "
            "Проверьте, запущен ли сервер."
        )

    except requests.exceptions.Timeout:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error("Время ожидания обработки истекло.")

    except requests.exceptions.HTTPError as error:
        shutil.rmtree(result_dir, ignore_errors=True)

        try:
            detail = error.response.json().get("detail", str(error))
        except (ValueError, AttributeError):
            detail = str(error)

        raise gr.Error(f"Ошибка FastAPI: {detail}")

    except requests.exceptions.RequestException as error:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error(f"Ошибка при обращении к API: {error}")

    except (zipfile.BadZipFile, OSError) as error:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error(
            f"Не удалось обработать результаты сервера: {error}"
        )

    except Exception as error:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error(f"Ошибка обработки: {error}")

LANGUAGE_CHOICES = [
    ("Оригинальный язык", "original"),
    ("Русский", "ru"),
    ("Английский", "en"),
    ("Немецкий", "de"),
    ("Французский", "fr"),
    ("Испанский", "es"),
    ("Китайский (упрощённый)", "zh")
]


def add_interval(intervals):
    intervals = list(intervals)
    intervals.append({
        "start": "00:00:00",
        "end": "00:01:00",
        "language": "ru"
    })
    return intervals


def remove_interval(intervals, index):
    intervals = list(intervals)

    if len(intervals) > 1:
        intervals.pop(index)

    return intervals


def update_interval(start, end, language, intervals, index):
    intervals = list(intervals)

    intervals[index] = {
        "start": start,
        "end": end,
        "language": language
    }

    return intervals


def update_subtitle_mode(mode):
    return (
        gr.update(visible=(mode == "single")),
        gr.update(visible=(mode == "multiple"))
    )

with gr.Blocks(title="SubtitleGenerator") as demo:

    intervals_state = gr.State([
        {
            "start": "00:00:00",
            "end": "00:01:30",
            "language": "ru"
        },
        {
            "start": "00:01:30",
            "end": "00:03:00",
            "language": "en"
        }
    ])

    gr.Markdown(
        """
        # SubtitleGenerator

        ### Автоматическая генерация субтитров для аудио и видео

        Загрузите аудио- или видеофайл, чтобы распознать речь
        и создать субтитры.

        Для видео субтитры также будут встроены
        непосредственно в видеоролик.
        """
    )

    file_input = gr.File(
        label="Выберите аудио- или видеофайл",
        type="filepath"
    )

    model_input = gr.Dropdown(
        choices=["small", "medium"],
        value="small",
        label="Модель распознавания Whisper",
        info="small — быстрее, medium — потенциально точнее",
        interactive=True
    )

    language_input = gr.Dropdown(
        choices=[
            ("Оригинальный язык (без перевода)", "original"),
            ("Русский", "ru"),
            ("Английский", "en"),
            ("Немецкий", "de"),
            ("Французский", "fr"),
            ("Испанский", "es"),
            ("Китайский (упрощённый)", "zh")
        ],
        value="original",
        label="Язык субтитров",
        info="Выберите язык, на котором должны быть созданы субтитры",
        interactive=True
    )

    subtitle_mode = gr.Radio(
        choices=[
            ("Один язык для всего видео", "single"),
            ("Несколько языков по интервалам", "multiple")
        ],
        value="single",
        label="Режим создания субтитров",
        interactive=True
    )

    censor_input = gr.Checkbox(
        label="Цензурировать ненормативную лексику",
        value=False,
        info="Заменять ненормативные слова на *** и звуковой сигнал"
    )

    generate_button = gr.Button(
        "Создать субтитры",
        variant="primary"
    )

    status = gr.Textbox(
        label="Статус обработки",
        interactive=False
    )

    video_output = gr.Video(
        label="Видео с субтитрами"
    )

    srt_output = gr.File(
        label="Скачать файл субтитров (SRT)"
    )

    # Настройка языковых интервалов находится внизу интерфейса.
    with gr.Column(visible=False) as intervals_container:

        gr.Markdown("### Настройка языковых интервалов")

        add_interval_button = gr.Button("+ Добавить интервал")

        @gr.render(inputs=intervals_state)
        def render_intervals(intervals):

            for index, interval in enumerate(intervals):

                with gr.Row():

                    start_input = gr.Textbox(
                        label="Начало",
                        value=interval["start"],
                        placeholder="00:00:00",
                        interactive = True
                    )

                    end_input = gr.Textbox(
                        label="Конец",
                        value=interval["end"],
                        placeholder="00:01:00",
                        interactive=True
                    )

                    language_input_interval = gr.Dropdown(
                        choices=LANGUAGE_CHOICES,
                        value=interval["language"],
                        label="Язык",
                        interactive=True
                    )

                    remove_button = gr.Button(
                        "Удалить",
                        size="sm"
                    )

                for component in [
                    start_input,
                    end_input,
                    language_input_interval
                ]:
                    component.change(
                        fn=lambda start, end, language, current, i=index:
                        update_interval(
                            start,
                            end,
                            language,
                            current,
                            i
                        ),
                        inputs=[
                            start_input,
                            end_input,
                            language_input_interval,
                            intervals_state
                        ],
                        outputs=intervals_state
                    )

                remove_button.click(
                    fn=lambda current, i=index:
                    remove_interval(current, i),
                    inputs=intervals_state,
                    outputs=intervals_state
                )

    # Обработчик добавления интервала регистрируется один раз.
    add_interval_button.click(
        fn=add_interval,
        inputs=intervals_state,
        outputs=intervals_state
    )

    # Переключение между режимами.
    subtitle_mode.change(
        fn=update_subtitle_mode,
        inputs=subtitle_mode,
        outputs=[
            language_input,
            intervals_container
        ]
    )

    # Запуск генерации субтитров.
    generate_button.click(
        fn=generate_subtitles,
        inputs=[
            file_input,
            model_input,
            language_input,
            censor_input,
            subtitle_mode,
            intervals_state
        ],
        outputs=[
            video_output,
            srt_output,
            status
        ]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860
    )