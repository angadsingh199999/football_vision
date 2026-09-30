from ultralytics import YOLO
import cv2
from pathlib import Path


# ==================================================
# PATHS
# ==================================================

MODEL_PATH = Path("models/best.pt")
VIDEO_PATH = Path("data/input/match.mp4")

OUTPUT_DIR = Path("outputs/tracking")
OUTPUT_VIDEO = OUTPUT_DIR / "ball_tracking_test_1280.mp4"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD MODEL
# ==================================================

model = YOLO(str(MODEL_PATH))

print("Model loaded")
print("Classes:", model.names)


# ==================================================
# OPEN VIDEO
# ==================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("================================")
print("BALL TRACKING TEST")
print("================================")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Test duration: 30 seconds")
print("Image size: 1280")


# ==================================================
# OUTPUT VIDEO
# ==================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


# ==================================================
# PROCESS 30 SECONDS
# ==================================================

max_frames = int(fps * 30)

frame_count = 0
ball_detections = 0
ball_frames = set()


while frame_count < max_frames:

    ret, frame = cap.read()

    if not ret:
        break

    # ----------------------------------------------
    # YOLO + BYTE TRACK
    # ----------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.20,
        imgsz=1280,
        verbose=False
    )

    result = results[0]

    # ----------------------------------------------
    # COUNT BALL DETECTIONS
    # ----------------------------------------------

    if result.boxes is not None:

        for i in range(len(result.boxes)):

            class_id = int(result.boxes.cls[i])

            if class_id == 0:

                ball_detections += 1
                ball_frames.add(frame_count)

    # ----------------------------------------------
    # SAVE VIDEO
    # ----------------------------------------------

    annotated_frame = result.plot()

    writer.write(annotated_frame)

    frame_count += 1

    if frame_count % 100 == 0:

        print(
            f"Processed "
            f"{frame_count}/{max_frames}"
        )


# ==================================================
# CLEANUP
# ==================================================

cap.release()
writer.release()


# ==================================================
# RESULTS
# ==================================================

coverage = (
    len(ball_frames) / frame_count * 100
    if frame_count > 0
    else 0
)

print("================================")
print("BALL TRACKING TEST COMPLETE")
print("================================")
print("Frames processed:", frame_count)
print("Ball detections:", ball_detections)
print("Ball frames:", len(ball_frames))
print("Ball coverage:", round(coverage, 2), "%")
print("Saved:", OUTPUT_VIDEO)
print("================================")