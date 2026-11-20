from transcriber import transcribe_video

VIDEO_PATH = "data/Video.mp4"

result = transcribe_video(VIDEO_PATH, model_name="small")

for segment in result["segments"]:
    print(f"\nСегмент: {segment['text']}")

    for word in segment.get("words", []):
        print(
            f"Слово: {word['word']!r} | "
            f"Начало: {word['start']:.2f} | "
            f"Конец: {word['end']:.2f}"
        )