from ultralytics import YOLO
from pathlib import Path


MODEL_PATH = "models/best.pt"
IMAGE_PATH = "data/frames/reference.jpg"

OUTPUT_DIR = Path("outputs/detection_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load trained model
model = YOLO(MODEL_PATH)

print("Model loaded")
print("Classes:", model.names)


# Run detection
results = model.predict(
    source=IMAGE_PATH,
    imgsz=640,
    conf=0.25,
    save=True,
    project=str(OUTPUT_DIR),
    name="result",
    exist_ok=True
)


print("\nDetection complete.")

for result in results:

    if result.boxes is None:
        print("No objects detected.")
        continue

    print("Objects detected:", len(result.boxes))

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"{class_name}: "
            f"{confidence:.3f}"
        )