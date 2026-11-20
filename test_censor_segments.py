from censor import censor_segments

test_segments = [
    {
        "text": "Это обычная речь.",
        "start": 0.0,
        "end": 2.0,
        "words": [
            {"word": "Это", "start": 0.0, "end": 0.3},
            {"word": " обычная", "start": 0.3, "end": 0.9},
            {"word": " речь.", "start": 0.9, "end": 1.5},
        ],
    },
    {
        "text": "Это блядь тест.",
        "start": 2.0,
        "end": 4.0,
        "words": [
            {"word": "Это", "start": 2.0, "end": 2.3},
            {"word": " блядь", "start": 2.3, "end": 2.8},
            {"word": " тест.", "start": 2.8, "end": 3.2},
        ],
    },
{
    "text": "Это блядь проверка.",
    "start": 4.0,
    "end": 6.0,
    "words": [
        {"word": "Это", "start": 4.0, "end": 4.3},
        {"word": " бля", "start": 4.3, "end": 4.5},
        {"word": "дь", "start": 4.5, "end": 4.8},
        {"word": " проверка.", "start": 4.8, "end": 5.5},
    ],
},
]

segments, intervals = censor_segments(test_segments, "ru")

for segment in segments:
    print(segment["text"])

print("Интервалы для сигналов:", intervals)
