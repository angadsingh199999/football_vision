import sys
import cv2
import json
from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from rapidocr import RapidOCR
from src.ocr.scoreboard_parser import parse_scoreboard


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = PROJECT_ROOT / "data/input/match.mp4"
CONFIG_PATH = PROJECT_ROOT / "config/scoreboard.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs/scoreboard_readings.json"


# ============================================================
# SETTINGS
# ============================================================

# Sample at a practical rate for full-length matches. Confirmed score changes
# require repeated readings, so a brief OCR miss does not create a goal.
SAMPLE_INTERVAL_SECONDS = 1.0

# Minimum OCR confidence.
MIN_CONFIDENCE = 0.50

# Upscale the very small scoreboard crop before OCR.
UPSCALE = 3


# ============================================================
# CHECK FILES
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Scoreboard configuration not found: {CONFIG_PATH}"
    )


# ============================================================
# LOAD SCOREBOARD COORDINATES
# ============================================================

with open(CONFIG_PATH, "r") as file:
    coordinates = json.load(file)

x1 = int(coordinates["x1"])
y1 = int(coordinates["y1"])
x2 = int(coordinates["x2"])
y2 = int(coordinates["y2"])
reference_width = int(coordinates.get("reference_width", 1280))
reference_height = int(coordinates.get("reference_height", 720))


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = (
    frame_count / fps
    if fps > 0
    else 0
)

print("=" * 70)
print("VIDEO SCOREBOARD OCR")
print("=" * 70)
print(f"Video          : {VIDEO_PATH}")
print(f"Duration       : {duration:.2f} seconds")
print(f"FPS            : {fps:.2f}")
print(f"Sampling       : every {SAMPLE_INTERVAL_SECONDS} second")
print(
    f"Scoreboard crop: "
    f"({x1}, {y1}) -> ({x2}, {y2}) at reference "
    f"{reference_width}x{reference_height}"
)
print(f"Upscale        : {UPSCALE}x")
print(f"OCR confidence : {MIN_CONFIDENCE}")
print("=" * 70)


# ============================================================
# INITIALIZE RAPIDOCR
# ============================================================

print()
print("Initializing RapidOCR...")

ocr = RapidOCR()

print("RapidOCR initialized.")
print()


def read_scoreboard_region(frame, region):
    """Run OCR on an absolute-pixel crop and parse the result."""
    rx1, ry1, rx2, ry2 = region
    height, width = frame.shape[:2]
    rx1 = max(0, min(int(rx1), width - 1))
    rx2 = max(rx1 + 1, min(int(rx2), width))
    ry1 = max(0, min(int(ry1), height - 1))
    ry2 = max(ry1 + 1, min(int(ry2), height))

    crop = frame[ry1:ry2, rx1:rx2]
    crop = cv2.resize(
        crop, None, fx=UPSCALE, fy=UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    try:
        result = ocr(crop)
    except Exception as exc:
        print(f"OCR ERROR for region {region}: {exc}")
        result = None

    raw_texts = []
    raw_scores = []
    valid_texts = []
    if result is not None and getattr(result, "txts", None):
        for text, confidence in zip(
            result.txts, getattr(result, "scores", ()),
        ):
            text = str(text).strip()
            confidence = float(confidence)
            raw_texts.append(text)
            raw_scores.append(round(confidence, 3))
            if text and confidence >= MIN_CONFIDENCE:
                valid_texts.append(text)

    parsed = parse_scoreboard(valid_texts)
    has_score = (
        parsed.get("home_score") is not None
        and parsed.get("away_score") is not None
    )
    has_teams = bool(parsed.get("home_team") and parsed.get("away_team"))
    has_clock = parsed.get("clock") is not None
    # The clock is a useful broadcast-overlay fingerprint. Requiring it in
    # the automatic crop prevents ad-board text and shirt numbers from
    # masquerading as a score while the real overlay is absent.
    if not (has_score and has_teams and has_clock):
        parsed["home_score"] = None
        parsed["away_score"] = None
    return parsed, raw_texts, raw_scores, has_score and has_teams and has_clock


# ============================================================
# STORAGE
# ============================================================

readings = []

current_time = 0.0

sample_number = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

while current_time <= duration:

    sample_number += 1

    # --------------------------------------------------------
    # SEEK TO TIMESTAMP
    # --------------------------------------------------------

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        current_time * 1000
    )

    success, frame = cap.read()

    if not success:

        print(
            f"[{current_time:6.1f}s] "
            "Could not read frame"
        )

        current_time += SAMPLE_INTERVAL_SECONDS
        continue


    height, width = frame.shape[:2]
    scale_x = width / max(reference_width, 1)
    scale_y = height / max(reference_height, 1)
    configured_region = (
        round(x1 * scale_x), round(y1 * scale_y),
        round(x2 * scale_x), round(y2 * scale_y),
    )

    # Try the user-configured crop first. If it misses the board, fall back to
    # a broad top-left band, which covers the common broadcast layout while
    # still allowing the user to configure another location in the app.
    band_width = max(1, round(width * 0.55))
    band_height = max(1, round(height * 0.19))
    auto_regions = [("top-left", (0, 0, band_width, band_height))]
    regions = [("configured", configured_region)] + auto_regions
    attempted = set()
    best_result = None
    scoreboard_region = "configured"

    for region_name, region in regions:
        if region in attempted:
            continue
        attempted.add(region)
        candidate = read_scoreboard_region(frame, region)
        parsed, candidate_texts, candidate_scores, is_valid = candidate

        # Keep the most informative crop for diagnostics when no full
        # scoreboard can be read. Prefer a parsed clock/labels to ad text.
        quality = (
            int(parsed.get("clock") is not None)
            + int(parsed.get("home_team") is not None)
            + int(parsed.get("away_team") is not None)
            + len(candidate_texts) / 100.0
        )
        if best_result is None or quality > best_result[0]:
            best_result = (
                quality, parsed, candidate_texts, candidate_scores, region_name,
            )

        if is_valid:
            best_result = (
                float("inf"), parsed, candidate_texts, candidate_scores, region_name,
            )
            break

    _, scoreboard, raw_texts, raw_scores, scoreboard_region = best_result


    # --------------------------------------------------------
    # CREATE READING
    # --------------------------------------------------------

    reading = {
        "video_time": round(current_time, 2),

        "clock": scoreboard.get("clock"),

        "home_team": scoreboard.get("home_team"),

        "home_score": scoreboard.get("home_score"),

        "away_score": scoreboard.get("away_score"),

        "away_team": scoreboard.get("away_team"),

        "scoreboard_region": scoreboard_region,

        # Keep OCR information for debugging.
        "ocr_texts": raw_texts,

        "ocr_confidences": raw_scores,
    }

    readings.append(reading)


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    score_text = (
        f"{scoreboard.get('home_score')}"
        f"-"
        f"{scoreboard.get('away_score')}"
    )

    print(
        f"[{current_time:6.1f}s] "
        f"clock={scoreboard.get('clock')} | "
        f"score={score_text} | "
        f"teams="
        f"{scoreboard.get('home_team')} "
        f"vs "
        f"{scoreboard.get('away_team')} | "
        f"OCR={raw_texts}"
    )


    # --------------------------------------------------------
    # NEXT SAMPLE
    # --------------------------------------------------------

    current_time += SAMPLE_INTERVAL_SECONDS


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w"
) as file:

    json.dump(
        readings,
        file,
        indent=4
    )


# ============================================================
# SUMMARY
# ============================================================

valid_scores = [
    r
    for r in readings
    if isinstance(r.get("home_score"), int)
    and isinstance(r.get("away_score"), int)
]


print()
print("=" * 70)
print("VIDEO OCR COMPLETED")
print("=" * 70)
print(f"Total readings       : {len(readings)}")
print(f"Valid score readings : {len(valid_scores)}")
print(f"Output               : {OUTPUT_PATH}")
print("=" * 70)
