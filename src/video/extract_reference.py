import cv2
from pathlib import Path


VIDEO_PATH = Path("data/input/match.mp4")
OUTPUT_PATH = Path("data/frames/reference.jpg")

# Time in seconds from the beginning of the video
TIME_SECONDS = 30


cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


# Get FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Convert seconds to frame number
target_frame = int(TIME_SECONDS * fps)

# Jump directly to that frame
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

success, frame = cap.read()

if not success:
    cap.release()
    raise RuntimeError("Could not read the reference frame.")


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(OUTPUT_PATH), frame)

print(f"Reference frame saved to: {OUTPUT_PATH}")
print(f"Time: {TIME_SECONDS} seconds")
print(f"Frame: {target_frame}")

cap.release()