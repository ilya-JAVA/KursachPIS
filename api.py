import logging
import shutil
import tempfile
import uuid
import zipfile

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from main import process_video, process_audio


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SubtitleGenerator API",
    description="API для автоматической генерации субтитров с помощью Whisper",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    """Проверка работоспособности API."""
    return {
        "status": "ok",
        "message": "SubtitleGenerator API is running"
    }


@app.post("/transcribe")
async def transcribe_video(
    file: UploadFile = File(...),
    model_name: str = Form("small"),
    target_language: str = Form("original"),
    censor_profanity: bool = Form(False)
):
    """Распознаёт речь, создаёт SRT и обрабатывает аудио и видео."""

    video_extensions = {".mp4", ".mov", ".mkv", ".avi"}
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}

    allowed_extensions = video_extensions | audio_extensions
    allowed_models = {"small", "medium"}

    allowed_languages = {
        "original", "ru", "en", "de", "fr", "es", "zh"
    }

    if target_language not in allowed_languages:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый язык перевода."
        )

    if model_name not in allowed_models:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемая модель. Доступны: small, medium."
        )

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Имя файла не указано."
        )

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат файла."
        )

    request_id = str(uuid.uuid4())

    temp_dir = Path(tempfile.mkdtemp())
    input_path = temp_dir / f"{request_id}{extension}"

    generated_files = [
        Path("output") / f"{request_id}_original_subtitled.mp4",
        Path("output") / f"{request_id}_ru_subtitled.mp4",
        Path("output") / f"{request_id}_en_subtitled.mp4",
        Path("output") / f"{request_id}_de_subtitled.mp4",
        Path("output") / f"{request_id}_fr_subtitled.mp4",
        Path("output") / f"{request_id}_es_subtitled.mp4",
        Path("output") / f"{request_id}_zh_subtitled.mp4",
        Path("output") / f"{request_id}_{model_name}_{target_language}.srt",
        Path("output") / (
            f"{request_id}_{model_name}_{target_language}_censored.wav"
        )
    ]

    try:
        logger.info(
            "Получен файл: %s. ID запроса: %s",
            filename,
            request_id
        )

        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if input_path.stat().st_size == 0:
            raise HTTPException(
                status_code=400,
                detail="Загруженный файл пуст."
            )

        logger.info("Начинаем обработку: %s", request_id)

        output_audio = None

        if extension in video_extensions:
            output_video, output_srt = process_video(
                str(input_path),
                model_name=model_name,
                target_language=target_language,
                censor_profanity=censor_profanity
            )
        else:
            output_srt, output_audio = process_audio(
                str(input_path),
                model_name=model_name,
                target_language=target_language,
                censor_profanity=censor_profanity
            )
            output_video = None

        zip_path = temp_dir / "subtitle_results.zip"

        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED
        ) as archive:

            if output_video is not None:
                archive.write(output_video, Path(output_video).name)

            archive.write(output_srt, Path(output_srt).name)

            if output_audio is not None:
                archive.write(output_audio, Path(output_audio).name)

        logger.info("Обработка успешно завершена: %s", request_id)

        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename="subtitle_results.zip",
            background=BackgroundTask(
                shutil.rmtree,
                temp_dir,
                ignore_errors=True
            )
        )

    except HTTPException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    except Exception:
        logger.exception(
            "Ошибка при обработке файла. ID запроса: %s",
            request_id
        )

        shutil.rmtree(temp_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail="Произошла ошибка при обработке файла."
        )

    finally:
        for generated_file in generated_files:
            generated_file.unlink(missing_ok=True)

        await file.close()
