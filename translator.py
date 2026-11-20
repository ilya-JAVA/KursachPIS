
def translate_segments(
    segments: list,
    source_language: str,
    target_language: str
) -> list:
    """Переводит сегменты Whisper, сохраняя временные метки и маркеры цензуры."""

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

        # Защищаем маркеры цензуры от изменений переводчиком
        placeholder = "CENSORPLACEHOLDER"
        protected_text = text.replace("***", placeholder)

        inputs = tokenizer(
            protected_text,
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

        # Возвращаем маркер цензуры
        translated_text = translated_text.replace(
            placeholder,
            "***"
        )

        translated_segment = segment.copy()
        translated_segment["text"] = translated_text
        translated_segments.append(translated_segment)

    return translated_segments
