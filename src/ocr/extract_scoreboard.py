import cv2
import json
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

IMAGE_PATH = Path("data/frames/reference.jpg")
CONFIG_PATH = Path("config/scoreboard.json")
OUTPUT_PATH = Path("data/frames/scoreboard_crop.jpg")


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise RuntimeError(f"Could not read image: {IMAGE_PATH}")


# --------------------------------------------------
# Load scoreboard coordinates
# --------------------------------------------------

with open(CONFIG_PATH, "r") as file:
    coordinates = json.load(file)


x1 = coordinates["x1"]
y1 = coordinates["y1"]
x2 = coordinates["x2"]
y2 = coordinates["y2"]


# --------------------------------------------------
# Validate coordinates
# --------------------------------------------------

height, width = image.shape[:2]

if not (0 <= x1 < x2 <= width):
    raise ValueError("Invalid X coordinates.")

if not (0 <= y1 < y2 <= height):
    raise ValueError("Invalid Y coordinates.")


# --------------------------------------------------
# Crop scoreboard
# --------------------------------------------------

scoreboard = image[y1:y2, x1:x2]


# --------------------------------------------------
# Save crop
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(OUTPUT_PATH), scoreboard)


print("Scoreboard crop created successfully.")
print(f"Input image : {IMAGE_PATH}")
print(f"Coordinates : ({x1}, {y1}) → ({x2}, {y2})")
print(f"Output      : {OUTPUT_PATH}")
print(f"Crop size   : {scoreboard.shape[1]} x {scoreboard.shape[0]}")