from pathlib import Path
import cv2
import random


DATASET = Path(
    "/Users/angadchhabra/Desktop/football_vision_dataset"
)

IMAGE_DIR = DATASET / "valid" / "images"
LABEL_DIR = DATASET / "valid" / "labels"


# Change this if your dataset is stored elsewhere.
# The script only reads the dataset.


images = list(IMAGE_DIR.glob("*"))

if not images:
    raise RuntimeError(
        f"No images found in: {IMAGE_DIR}"
    )


random.seed(42)

selected = random.sample(
    images,
    min(10, len(images))
)


OUTPUT_DIR = Path(
    "outputs/dataset_check"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for image_path in selected:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        continue

    height, width = image.shape[:2]

    label_path = (
        LABEL_DIR /
        f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        continue

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            x1 = int(
                (x_center - box_width / 2)
                * width
            )

            y1 = int(
                (y_center - box_height / 2)
                * height
            )

            x2 = int(
                (x_center + box_width / 2)
                * width
            )

            y2 = int(
                (y_center + box_height / 2)
                * height
            )

            if class_id == 0:
                label = "BALL"
            elif class_id == 1:
                label = "PLAYER"
            else:
                continue

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

    output_path = (
        OUTPUT_DIR /
        image_path.name
    )

    cv2.imwrite(
        str(output_path),
        image
    )

    print(
        "Saved:",
        output_path
    )


print("\nDataset visualization complete.")