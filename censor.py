import re
from pathlib import Path

# Словари ненормативной лексики
PROFANITY_WORDS = {
    "ru": {
        "блядь",
        "блять",
        "сука",
        "хуй",
        "пизда",
        "ебать",
        "ебал",
        "ебёт",
        "мудак",
        "гандон",
        "долбоеб",
        "далбаеб",
        "пидор",
        "пидар",
        "пидорас",
        "пидарас",
        "нахуй",
        "ебучая",
    },
    "en": {
        "fuck",
        "fucking",
        "fucked",
        "shit",
        "bitch",
        "asshole",
        "bastard",
        "dick",
        "motherfucker",
    },
    "de": {
        "scheiße",
        "fick",
        "ficken",
        "arschloch",
        "hurensohn",
    },
    "fr": {
        "merde",
        "putain",
        "connard",
        "salope",
        "foutre",
    },
    "es": {
        "mierda",
        "joder",
        "puta",
        "puto",
        "coño",
        "cabrón",
    },
    "zh": {
        "他妈的",
        "傻逼",
        "混蛋",
        "操",
    },
}


def normalize_word(word: str) -> str:
    """Нормализует слово для сравнения со словарём."""
    return re.sub(r"[^\w]", "", word.casefold(), flags=re.UNICODE)



def censor_segments(segments: list, language: str) -> tuple[list, list]:
    """
    Заменяет ненормативные слова на ***.
    Учитывает слова, разделённые Whisper на несколько фрагментов.
    Возвращает изменённые сегменты и интервалы для звуковых сигналов.
    """
    if language not in PROFANITY_WORDS:
        raise ValueError(f"Язык не поддерживается для цензурирования: {language}")

    censored_segments = []
    beep_intervals = []
    profanity = PROFANITY_WORDS[language]

    for segment in segments:
        censored_segment = segment.copy()
        words = segment.get("words", [])

        if not words:
            censored_segments.append(censored_segment)
            continue

        censored_words = []
        i = 0

        while i < len(words):
            current_word = words[i]["word"]
            leading_space = current_word[:len(current_word) - len(current_word.lstrip())]

            matched = False

            # Проверяем последовательности из нескольких фрагментов
            for count in range(1, min(4, len(words) - i) + 1):
                group = words[i:i + count]

                # Не объединяем разные слова, если следующий фрагмент
                # начинается с пробела
                if count > 1 and any(
                    item["word"].startswith((" ", "\t", "\n"))
                    for item in group[1:]
                ):
                    break

                combined = "".join(item["word"] for item in group)
                normalized = normalize_word(combined)

                if normalized in profanity:
                    censored_words.append(leading_space + "***")

                    start = min(float(item["start"]) for item in group)
                    end = max(float(item["end"]) for item in group)

                    beep_intervals.append((start, end))
                    i += count
                    matched = True
                    break

            if not matched:
                censored_words.append(current_word)
                i += 1

        censored_segment["text"] = "".join(censored_words)
        censored_segments.append(censored_segment)

    return censored_segments, beep_intervals



from pydub import AudioSegment
from pydub.generators import Sine




def censor_audio(
    audio_path: str,
    beep_intervals: list,
    output_path: str
) -> str:
    """Заменяет ненормативную речь звуковыми сигналами."""

    audio = AudioSegment.from_file(audio_path)

    beep_duration = 300

    beep = Sine(1000).to_audio_segment(
        duration=beep_duration
    )

    beep = (beep - 5).set_frame_rate(audio.frame_rate)
    beep = beep.set_channels(audio.channels)
    beep = beep.set_sample_width(audio.sample_width)

    prepared_intervals = []

    # Подготавливаем интервалы для сигналов
    for start, end in beep_intervals:
        start_ms = max(0, int(start * 1000))
        end_ms = min(len(audio), int(end * 1000))

        if start_ms >= len(audio) or end_ms <= start_ms:
            continue

        word_center = (start_ms + end_ms) // 2

        beep_start = word_center - beep_duration // 2
        beep_end = beep_start + beep_duration

        # Корректируем границы около начала и конца аудио
        if beep_start < 0:
            beep_start = 0
            beep_end = min(beep_duration, len(audio))

        if beep_end > len(audio):
            beep_end = len(audio)
            beep_start = max(0, beep_end - beep_duration)

        prepared_intervals.append((beep_start, beep_end))

    # Сортируем интервалы
    prepared_intervals.sort()

    # Объединяем пересекающиеся интервалы
    merged_intervals = []

    for start, end in prepared_intervals:
        if merged_intervals and start <= merged_intervals[-1][1]:
            merged_intervals[-1] = (
                merged_intervals[-1][0],
                max(merged_intervals[-1][1], end)
            )
        else:
            merged_intervals.append((start, end))

    # Заглушаем участки под сигналами
    for start, end in merged_intervals:
        silence = AudioSegment.silent(
            duration=end - start,
            frame_rate=audio.frame_rate
        )

        silence = silence.set_channels(audio.channels)
        silence = silence.set_sample_width(audio.sample_width)

        audio = audio[:start] + silence + audio[end:]

    # Накладываем звуковые сигналы
    for start, end in merged_intervals:
        beep_part = beep[:end - start]

        audio = audio.overlay(
            beep_part,
            position=start
        )

    # Сохраняем результат
    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    audio.export(output_path, format="wav")

    return output_path
