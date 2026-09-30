import os
import cv2
import pandas as pd


VIDEO_FILE = "data/input/match.mp4"
EVENT_FILE = "outputs/events/final_goal_save_events.csv"

OUTPUT_DIR = "outputs/highlights/saves"


def extract_clip(
    cap,
    fps,
    start_time,
    end_time,
    output_file
):
    start_frame = max(
        0,
        int(start_time * fps)
    )

    end_frame = int(
        end_time * fps
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    writer = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {output_file}"
        )

    frame_no = start_frame

    while frame_no <= end_frame:

        ret, frame = cap.read()

        if not ret:
            break

        writer.write(frame)

        frame_no += 1

    writer.release()


print("=" * 70)
print("SAVE CLIP EXTRACTION")
print("=" * 70)


if not os.path.exists(VIDEO_FILE):
    raise FileNotFoundError(
        f"Video not found: {VIDEO_FILE}"
    )

if not os.path.exists(EVENT_FILE):
    raise FileNotFoundError(
        f"Event file not found: {EVENT_FILE}"
    )


events = pd.read_csv(EVENT_FILE)

saves = events[
    events["event_type"] == "Save"
].copy()

saves = saves.sort_values(
    "event_time"
).reset_index(drop=True)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_FILE}"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

print(f"FPS: {fps}")
print(f"Saves found: {len(saves)}")
print()


for save_number, (_, row) in enumerate(
    saves.iterrows(),
    start=1
):

    event_time = float(
        row["event_time"]
    )

    start_time = float(
        row["start_time"]
    )

    end_time = float(
        row["end_time"]
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"save_{save_number}_{event_time:.2f}s.mp4"
    )

    print(
        f"Save {save_number}: "
        f"{start_time:.2f}s -> "
        f"{end_time:.2f}s"
    )

    extract_clip(
        cap,
        fps,
        start_time,
        end_time,
        output_file
    )

    print(
        f"  Saved: {output_file}"
    )


cap.release()


print()
print("=" * 70)
print("SAVE CLIPS COMPLETE")
print("=" * 70)
