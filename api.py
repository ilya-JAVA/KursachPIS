import logging
import shutil
import tempfile
import uuid
import zipfile

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from main import process_video


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Создание FastAPI-приложения
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
async def transcribe_video(file: UploadFile = File(...)):
    """Распознаёт речь, создаёт SRT и встраивает субтитры в видео."""

    allowed_extensions = {".mp4", ".mov", ".mkv", ".avi"}

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Неподдерживаемый формат файла."
        )

    request_id = str(uuid.uuid4())

    temp_dir = Path(tempfile.mkdtemp())
    input_path = temp_dir / f"{request_id}{extension}"

    try:
        logger.info("Получен файл: %s", file.filename)

        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info("Начинаем обработку файла: %s", file.filename)

        output_video, output_srt = process_video(str(input_path))

        zip_path = temp_dir / "subtitle_results.zip"

        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(output_video, Path(output_video).name)
            archive.write(output_srt, Path(output_srt).name)

        logger.info("Обработка завершена: %s", file.filename)

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

    except Exception:
        logger.exception("Ошибка при обработке файла")

        shutil.rmtree(temp_dir, ignore_errors=True)

        raise HTTPException(
            status_code=500,
            detail="Произошла ошибка при обработке видео."
        )

    finally:
        await file.close()