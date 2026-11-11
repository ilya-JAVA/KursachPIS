import shutil
import uuid
import zipfile

from pathlib import Path

import gradio as gr
import requests


API_URL = "http://127.0.0.1:8000/transcribe"

# Папка для результатов, полученных через интерфейс
RESULTS_DIR = Path("output/gradio_results")


def generate_subtitles(video_path, model_name):
    """Отправляет видео в FastAPI и получает готовые файлы."""

    if not video_path:
        raise gr.Error("Сначала выберите видеофайл.")

    video = Path(video_path)

    if not video.is_file():
        raise gr.Error("Выбранный видеофайл не найден.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_dir = RESULTS_DIR / str(uuid.uuid4())
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Отправляем видео в FastAPI
        with video.open("rb") as video_file:
            response = requests.post(
                API_URL,
                files={
                    "file": (
                        video.name,
                        video_file,
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
            archive.extractall(result_dir)

        # Находим готовые файлы
        output_video = next(
            result_dir.glob("*_subtitled.mp4"),
            None
        )

        output_srt = next(
            result_dir.glob("*.srt"),
            None
        )

        if output_video is None or output_srt is None:
            raise gr.Error(
                "Сервер не вернул видео или файл субтитров."
            )

        return (
            str(output_video),
            str(output_srt),
            "Субтитры успешно созданы!"
        )

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

    except requests.exceptions.RequestException as error:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error(f"Ошибка при обращении к API: {error}")

    except Exception as error:
        shutil.rmtree(result_dir, ignore_errors=True)
        raise gr.Error(f"Ошибка обработки: {error}")


# Создание интерфейса
with gr.Blocks(title="SubtitleGenerator") as demo:

    gr.Markdown(
        """
        # SubtitleGenerator

        ### Автоматическая генерация субтитров для видео

        Загрузите видео, чтобы распознать речь,
        создать субтитры и встроить их в видео.
        """
    )

    video_input = gr.File(
        label="Выберите видео",
        file_types=[".mp4", ".mov", ".mkv", ".avi"],
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
        inputs=video_input,
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
