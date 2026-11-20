import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "facebook/nllb-200-distilled-600M"

TEST_CASES = [
    {
        "source_language": "rus_Cyrl",
        "target_language": "eng_Latn",
        "source_name": "Русский",
        "target_name": "Английский",
        "text": "Я рад тебя видеть."
    },
    {
        "source_language": "deu_Latn",
        "target_language": "rus_Cyrl",
        "source_name": "Немецкий",
        "target_name": "Русский",
        "text": "Ich freue mich, dich zu sehen."
    },
    {
        "source_language": "fra_Latn",
        "target_language": "rus_Cyrl",
        "source_name": "Французский",
        "target_name": "Русский",
        "text": "Je suis content de vous voir."
    },
    {
        "source_language": "spa_Latn",
        "target_language": "rus_Cyrl",
        "source_name": "Испанский",
        "target_name": "Русский",
        "text": "Me alegro de verte."
    },
    {
        "source_language": "zho_Hans",
        "target_language": "rus_Cyrl",
        "source_name": "Китайский",
        "target_name": "Русский",
        "text": "我很高兴见到你。"
    }
]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Устройство: {device}")
    print("Загружаем модель и токенизатор...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()
    model.generation_config.max_length = None

    print("Модель успешно загружена!")

    for test in TEST_CASES:
        print("\n" + "=" * 55)
        print(
            f"Перевод: {test['source_name']} "
            f"-> {test['target_name']}"
        )
        print(f"Исходный текст: {test['text']}")

        tokenizer.src_lang = test["source_language"]

        inputs = tokenizer(
            test["text"],
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        target_language_id = tokenizer.convert_tokens_to_ids(
            test["target_language"]
        )

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

        print(f"Перевод: {translated_text}")

    print("\n" + "=" * 55)
    print("Тестирование завершено!")


if __name__ == "__main__":
    main()
