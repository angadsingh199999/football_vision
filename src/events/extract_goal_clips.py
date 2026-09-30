import os
import shutil
import subprocess
import cv2
import pandas as pd

VIDEO_FILE = "data/input/match.mp4"
EVENT_FILE = "outputs/events/final_goal_events.csv"
OUTPUT_DIR = "outputs/highlights/goals"


def extract_clip(cap, fps, start_time, end_time, output_file):
    start_frame = max(0, int(start_time * fps))
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_file}")

    frame_no = start_frame

    while frame_no <= end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        writer.write(frame)
        frame_no += 1

    writer.release()


print("=" * 70)
print("GOAL CLIP EXTRACTION")
print("=" * 70)

FFMPEG = shutil.which("ffmpeg")
if FFMPEG is None:
    try:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "FFmpeg is required to encode browser-playable H.264 goal clips. "
            "Install ffmpeg or the imageio-ffmpeg package."
        ) from exc

if not os.path.exists(VIDEO_FILE):
    raise FileNotFoundError(f"Video not found: {VIDEO_FILE}")

if not os.path.exists(EVENT_FILE):
    raise FileNotFoundError(f"Event file not found: {EVENT_FILE}")

events = pd.read_csv(EVENT_FILE)

if events.empty:
    print("No goal events found.")
    raise SystemExit(0)

# Safety check: this script must only process Goal events.
goals = events[events["event_type"] == "Goal"].copy()

goals = goals.sort_values("event_time").reset_index(drop=True)

# Remove old goal clips so stale incorrect clips do not remain.
os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith(".mp4"):
        os.remove(os.path.join(OUTPUT_DIR, filename))

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_FILE}")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    cap.release()
    raise RuntimeError("Invalid FPS detected.")

print(f"Video FPS : {fps}")
print(f"Goals     : {len(goals)}")
print()

for goal_number, (_, row) in enumerate(goals.iterrows(), start=1):

    event_time = float(row["event_time"])
    start_time = float(row["start_time"])
    end_time = float(row["end_time"])

    output_file = os.path.join(
        OUTPUT_DIR,
        f"goal_{goal_number}_{event_time:.2f}s.mp4"
    )
    raw_file = output_file.replace(".mp4", ".raw.mp4")
    encoded_file = output_file.replace(".mp4", ".h264.mp4")

    print(
        f"Goal {goal_number}: "
        f"{start_time:.2f}s -> {end_time:.2f}s "
        f"(event {event_time:.2f}s)"
    )

    extract_clip(
        cap,
        fps,
        start_time,
        end_time,
        raw_file,
    )

    # OpenCV's mp4v output is readable by OpenCV but is not supported by many
    # browsers. Re-encode to H.264 with fast-start metadata for st.video.
    try:
        subprocess.run(
            [
                FFMPEG,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", raw_file,
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                encoded_file,
            ],
            check=True,
        )
        os.replace(encoded_file, output_file)
    finally:
        for temporary_file in (raw_file, encoded_file):
            if os.path.exists(temporary_file):
                os.remove(temporary_file)

    print(f"  Saved: {output_file}")

cap.release()

print()
print("GOAL CLIP EXTRACTION COMPLETE")
print("=" * 70)
