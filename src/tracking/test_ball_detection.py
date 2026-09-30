from ultralytics import YOLO
import cv2
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path("models/best.pt")
VIDEO_PATH = Path("data/input/match.mp4")

OUTPUT_DIR = Path("outputs/tracking")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "ball_detection_test.mp4"


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(str(MODEL_PATH))

print("Model loaded")
print("Classes:", model.names)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


# ============================================================
# TEST ONLY FIRST 30 SECONDS
# ============================================================

max_frames = min(
    total_frames,
    int(fps * 30)
)


print("================================")
print("BALL DETECTION TEST")
print("================================")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Test frames:", max_frames)
print("Image size:", 1280)
print("================================")


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# DETECTION
# ============================================================

frame_count = 0
ball_frames = 0
ball_detections = 0


while frame_count < max_frames:

    ret, frame = cap.read()

    if not ret:
        break


    results = model.predict(
        frame,
        imgsz=1280,
        conf=0.15,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------------
    # Draw detections
    # --------------------------------------------------------

    annotated = frame.copy()

    found_ball = False


    if result.boxes is not None:

        for box, cls, conf in zip(
            result.boxes.xyxy,
            result.boxes.cls,
            result.boxes.conf
        ):

            class_id = int(cls)
            confidence = float(conf)


            # ------------------------------------------------
            # Only visualize BALL
            # ------------------------------------------------

            if class_id != 0:
                continue


            found_ball = True
            ball_detections += 1


            x1, y1, x2, y2 = map(
                int,
                box.tolist()
            )


            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            cv2.putText(
                annotated,
                f"ball {confidence:.2f}",
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    if found_ball:
        ball_frames += 1


    writer.write(annotated)


    frame_count += 1


    if frame_count % 100 == 0:

        print(
            f"Processed {frame_count}/{max_frames}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()


coverage = (
    ball_frames / frame_count * 100
    if frame_count > 0
    else 0
)


print()
print("================================")
print("BALL DETECTION TEST COMPLETE")
print("================================")
print("Frames processed:", frame_count)
print("Ball detections:", ball_detections)
print("Ball frames:", ball_frames)
print(
    "Ball coverage:",
    round(coverage, 2),
    "%"
)
print("Saved:", OUTPUT_PATH)
print("================================")