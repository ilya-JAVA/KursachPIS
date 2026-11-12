import shutil
import uuid
import zipfile

from pathlib import Path

import gradio as gr
import requests


API_URL = "http://127.0.0.1:8000/transcribe"

# Папка для результатов, полученных через интерфейс
RESULTS_DIR = Path("output/gradio_results")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def generate_subtitles(file_path, model_name):
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_dir = RESULTS_DIR / str(uuid.uuid4())
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Отправляем файл в FastAPI
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
                    "model_name": model_name
                },
                timeout=1800
            )

        response.raise_for_status()

        # Сохраняем полученный ZIP-архив
        archive_path = result_dir / "results.zip"
        archive_path.write_bytes(response.content)

        # Распаковываем результаты
        with zipfile.ZipFile(archive_path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue

                # Берём только имя файла, без путей из архива
                filename = Path(member.filename).name
                file_extension = Path(filename).suffix.lower()

                if file_extension not in {".mp4", ".srt"}:
                    continue

                output_path = result_dir / filename

                with archive.open(member) as source:
                    with output_path.open("wb") as destination:
                        shutil.copyfileobj(source, destination)

        # Находим готовые файлы
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

        # Формируем сообщение в зависимости от типа файла
        if output_video is not None:
            status_message = (
                "Видео с субтитрами и файл SRT успешно созданы!"
            )
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
        raise gr.Error(
            "Время ожидания обработки истекло."
        )

    except requests.exceptions.HTTPError as error:
        shutil.rmtree(result_dir, ignore_errors=True)

        try:
            detail = error.response.json().get("detail", str(error))
        except (ValueError, AttributeError):
            detail = str(error)

        raise gr.Error(f"Ошибка FastAPI: {detail}")

    except requests.exceptions.RequestException as error:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error(
            f"Ошибка при обращении к API: {error}"
        )

    except (zipfile.BadZipFile, OSError) as error:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error(
            f"Не удалось обработать результаты сервера: {error}"
        )

    except Exception as error:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error(
            f"Ошибка обработки: {error}"
        )


# Создание интерфейса
with gr.Blocks(title="SubtitleGenerator") as demo:

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

    generate_button.click(
        fn=generate_subtitles,
        inputs=[file_input, model_input],
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
