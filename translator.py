import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Поддерживаемые языки: код Whisper -> код NLLB
LANGUAGES = {
    "ru": "rus_Cyrl",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
}

# Выбираем доступное устройство
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Модель и токенизатор будут загружены при первом переводе
_tokenizer = None
_model = None


def load_translation_model():
    """Загружает модель и токенизатор один раз."""
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print(f"Загрузка модели перевода на устройстве: {DEVICE}")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        _model.to(DEVICE)
        _model.eval()

    return _tokenizer, _model


def translate_segments(
    segments: list,
    source_language: str,
    target_language: str
) -> list:
    """Переводит текст сегментов Whisper, сохраняя временные метки."""

    if target_language == "original":
        return segments

    if target_language not in LANGUAGES:
        raise ValueError(
            f"Неподдерживаемый целевой язык: {target_language}"
        )

    if source_language not in LANGUAGES:
        raise ValueError(
            f"Не удалось определить или поддержать исходный язык: "
            f"{source_language}"
        )

    # Если исходный и целевой языки совпадают, перевод не нужен
    if source_language == target_language:
        return segments

    tokenizer, model = load_translation_model()

    tokenizer.src_lang = LANGUAGES[source_language]
    target_language_id = tokenizer.convert_tokens_to_ids(
        LANGUAGES[target_language]
    )

    translated_segments = []

    for segment in segments:
        text = segment["text"].strip()

        if not text:
            translated_segments.append(segment.copy())
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(DEVICE)

        with torch.inference_mode():
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=target_language_id,
                max_new_tokens=128
            )

        translated_text = tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]

        translated_segment = segment.copy()
        translated_segment["text"] = translated_text
        translated_segments.append(translated_segment)

    return translated_segments