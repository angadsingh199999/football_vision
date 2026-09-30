from ultralytics import YOLO
import cv2
import csv
from pathlib import Path


# ==================================================
# PATHS
# ==================================================

MODEL_PATH = Path("models/best.pt")
VIDEO_PATH = Path("data/input/match.mp4")

OUTPUT_DIR = Path("outputs/tracking")
OUTPUT_VIDEO = OUTPUT_DIR / "tracked_match.mp4"
OUTPUT_CSV = OUTPUT_DIR / "tracking_data.csv"

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
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("================================")
print("VIDEO")
print("================================")
print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Frames:", total_frames)
print("Duration:", total_frames / fps, "seconds")


# ==================================================
# VIDEO WRITER
# ==================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)


# ==================================================
# CSV FILE
# ==================================================

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline=""
)

csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame",
    "time",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y"
])


# ==================================================
# PROCESS VIDEO
# ==================================================

frame_number = 0

while True:

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
        conf=0.15,
        imgsz=1280,
        verbose=False
    )

    result = results[0]

    # ----------------------------------------------
    # SAVE DETECTIONS
    # ----------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            class_id = int(boxes.cls[i])
            confidence = float(boxes.conf[i])

            # ByteTrack ID
            if boxes.id is not None:
                track_id = int(boxes.id[i])
            else:
                track_id = -1

            x1, y1, x2, y2 = boxes.xyxy[i].tolist()

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            class_name = model.names[class_id]

            time_seconds = frame_number / fps

            csv_writer.writerow([
                frame_number,
                round(time_seconds, 3),
                track_id,
                class_id,
                class_name,
                round(confidence, 4),
                round(x1, 2),
                round(y1, 2),
                round(x2, 2),
                round(y2, 2),
                round(center_x, 2),
                round(center_y, 2)
            ])

    # ----------------------------------------------
    # SAVE ANNOTATED VIDEO
    # ----------------------------------------------

    annotated_frame = result.plot()

    writer.write(annotated_frame)

    frame_number += 1

    if frame_number % 250 == 0:

        print(
            f"Processed "
            f"{frame_number}/{total_frames} frames "
            f"({frame_number / fps:.1f}s)"
        )


# ==================================================
# CLEANUP
# ==================================================

cap.release()
writer.release()
csv_file.close()


print("================================")
print("TRACKING COMPLETE")
print("================================")
print("Frames processed:", frame_number)
print("Tracking video:", OUTPUT_VIDEO)
print("Tracking CSV:", OUTPUT_CSV)
print("================================")