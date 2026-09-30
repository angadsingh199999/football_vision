from ultralytics import YOLO
import cv2
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = Path(
    "models/best.pt"
)

VIDEO_PATH = Path(
    "data/input/match.mp4"
)

OUTPUT_PATH = Path(
    "outputs/tracking_test.mp4"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = YOLO(str(MODEL_PATH))


# --------------------------------------------------
# OPEN VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("================================")
print("TRACKING TEST")
print("================================")
print("Resolution:", width, "x", height)
print("FPS:", fps)


# --------------------------------------------------
# OUTPUT VIDEO
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)


# --------------------------------------------------
# PROCESS FIRST 30 SECONDS
# --------------------------------------------------

max_frames = int(fps * 30)

frame_count = 0


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
        conf=0.25,
        imgsz=640,
        verbose=False
    )

    # ----------------------------------------------
    # DRAW TRACKING RESULTS
    # ----------------------------------------------

    annotated_frame = results[0].plot()

    writer.write(annotated_frame)

    frame_count += 1

    if frame_count % 100 == 0:

        print(
            f"Processed frames: "
            f"{frame_count}/{max_frames}"
        )


# --------------------------------------------------
# CLEANUP
# --------------------------------------------------

cap.release()
writer.release()

print("================================")
print("TRACKING COMPLETE")
print("Frames:", frame_count)
print("Saved:", OUTPUT_PATH)
print("================================")