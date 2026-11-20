from pathlib import Path

from pydub import AudioSegment

from censor import censor_audio


# Исходный аудиофайл
AUDIO_PATH = "data/Video.mp4"

# Временные интервалы для проверки (секунды)
TEST_INTERVALS = [
    (2.5, 2.9),
    (5.0, 5.4),
]

# Результат
OUTPUT_PATH = "output/test_censored.wav"

Path("output").mkdir(exist_ok=True)

# Обрабатываем аудио
result_path = censor_audio(
    audio_path=AUDIO_PATH,
    beep_intervals=TEST_INTERVALS,
    output_path=OUTPUT_PATH
)

# Сравниваем длительность
original = AudioSegment.from_file(AUDIO_PATH)
processed = AudioSegment.from_file(result_path)

print(f"Исходная длительность: {len(original) / 1000:.3f} сек.")
print(f"Обработанная длительность: {len(processed) / 1000:.3f} сек.")
print(f"Разница: {abs(len(original) - len(processed))} мс")
print(f"Результат сохранён: {result_path}")
