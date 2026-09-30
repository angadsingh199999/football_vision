import cv2
from pathlib import Path


# --------------------------------------------------
# 1. Video path
# --------------------------------------------------

VIDEO_PATH = Path("data/input/match.mp4")


# --------------------------------------------------
# 2. Output directory for sample frames
# --------------------------------------------------

OUTPUT_DIR = Path("data/frames")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. Open the video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))


# --------------------------------------------------
# 4. Check whether the video opened successfully
# --------------------------------------------------

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


# --------------------------------------------------
# 5. Get video information
# --------------------------------------------------

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# Calculate duration

duration = frame_count / fps if fps > 0 else 0


# --------------------------------------------------
# 6. Print video information
# --------------------------------------------------

print("========== VIDEO INFORMATION ==========")
print(f"File       : {VIDEO_PATH}")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.2f}")
print(f"Frames     : {frame_count}")
print(f"Duration   : {duration:.2f} seconds")
print("========================================")


# --------------------------------------------------
# 7. Read the video frame by frame
# --------------------------------------------------

frame_number = 0

while True:

    success, frame = cap.read()

    # Stop when there are no more frames
    if not success:
        break

    frame_number += 1

    # ----------------------------------------------
    # Save every 300th frame as a sample
    # ----------------------------------------------

    if frame_number % 300 == 0:

        output_path = OUTPUT_DIR / f"frame_{frame_number}.jpg"

        cv2.imwrite(str(output_path), frame)

        print(f"Saved: {output_path}")


# --------------------------------------------------
# 8. Release the video
# --------------------------------------------------

cap.release()

print()
print("Video processing completed.")