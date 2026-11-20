
from translator import translate_segments


test_segments = [
    {
        "start": 0.0,
        "end": 2.5,
        "text": "Hello! How are you?"
    },
    {
        "start": 2.5,
        "end": 5.0,
        "text": "I am glad to see you."
    }
]

result = translate_segments(
    segments=test_segments,
    source_language="en",
    target_language="ru"
)

print("\nРезультат перевода:")

for segment in result:
    print(
        f"[{segment['start']} - {segment['end']}] "
        f"{segment['text']}"
    )
