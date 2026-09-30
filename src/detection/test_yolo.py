from ultralytics import YOLO


# --------------------------------------------------
# Load pretrained YOLO model
# --------------------------------------------------

model = YOLO("yolo11n.pt")


# --------------------------------------------------
# Run detection on reference frame
# --------------------------------------------------

results = model(
    "data/frames/reference.jpg",
    conf=0.25
)


# --------------------------------------------------
# Print detections
# --------------------------------------------------

for result in results:

    print("\n========== DETECTIONS ==========\n")

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(
            f"{class_name:15} "
            f"confidence={confidence:.3f} "
            f"bbox=("
            f"{x1:.1f}, "
            f"{y1:.1f}, "
            f"{x2:.1f}, "
            f"{y2:.1f})"
        )

    print("\n===============================")