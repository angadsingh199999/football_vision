from pathlib import Path

from rapidocr import RapidOCR

from src.ocr.scoreboard_parser import parse_scoreboard
# --------------------------------------------------
# Input scoreboard image
# --------------------------------------------------

IMAGE_PATH = Path("data/frames/scoreboard_crop.jpg")


# --------------------------------------------------
# Check image
# --------------------------------------------------

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Scoreboard image not found: {IMAGE_PATH}"
    )


# --------------------------------------------------
# Initialize RapidOCR
# --------------------------------------------------

ocr = RapidOCR()


# --------------------------------------------------
# Run OCR
# --------------------------------------------------

result = ocr(str(IMAGE_PATH))


# --------------------------------------------------
# Check OCR result
# --------------------------------------------------

if result is None or not result.txts:

    print("No text detected.")

else:

    # RapidOCR gives us the recognized text
    texts = result.txts

    print("\n========== RAW OCR ==========\n")

    for text, score in zip(result.txts, result.scores):
        print(f"{text}  | confidence: {score:.4f}")

    print("\n=============================")

    # --------------------------------------------------
    # Parse scoreboard
    # --------------------------------------------------

    scoreboard = parse_scoreboard(texts)

    print("\n======= SCOREBOARD DATA =======\n")

    for key, value in scoreboard.items():
        print(f"{key:12}: {value}")

    print("\n===============================")