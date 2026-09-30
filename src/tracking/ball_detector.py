from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

VIDEO_PATH = Path("data/input/match.mp4")
MODEL_PATH = Path("models/best.pt")

BALL_OUTPUT = Path("outputs/tracking/ball_detections.csv")
PLAYER_OUTPUT = Path("outputs/tracking/player_detections.csv")


# ============================================================
# MODEL / DETECTION SETTINGS
# ============================================================

IMG_SIZE = 1280
CONFIDENCE = 0.15

BALL_CLASS = 0
PLAYER_CLASS = 1

# We explicitly request both classes.
CLASSES = [BALL_CLASS, PLAYER_CLASS]


def main():

    print("=" * 70)
    print("YOLO BALL + PLAYER DETECTION")
    print("=" * 70)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    BALL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps}")
    print(f"Frames: {frame_count}")
    print(f"Image size: {width} x {height}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Image size for YOLO: {IMG_SIZE}")
    print(f"Classes: {CLASSES}")
    print()

    ball_rows = []
    player_rows = []

    processed = 0

    # stream=True avoids keeping the complete result list in memory.
    results = model.predict(
        source=str(VIDEO_PATH),
        stream=True,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        classes=CLASSES,
        verbose=False,
    )

    for result in results:

        frame_index = processed
        time_sec = frame_index / fps if fps > 0 else 0.0

        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for box, confidence, class_id in zip(
                xyxy,
                confs,
                classes
            ):

                x1, y1, x2, y2 = map(float, box)

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                row = {
                    "frame": frame_index,
                    "time": time_sec,
                    "confidence": float(confidence),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                }

                if class_id == BALL_CLASS:
                    ball_rows.append(row)

                elif class_id == PLAYER_CLASS:
                    player_rows.append(row)

        processed += 1

        if processed % 250 == 0:
            print(
                f"Processed {processed}/{frame_count} "
                f"({processed / max(frame_count, 1) * 100:.1f}%)"
            )

    cap.release()

    ball_df = pd.DataFrame(
        ball_rows,
        columns=[
            "frame",
            "time",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
        ],
    )

    player_df = pd.DataFrame(
        player_rows,
        columns=[
            "frame",
            "time",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "center_x",
            "center_y",
        ],
    )

    ball_df.to_csv(BALL_OUTPUT, index=False)
    player_df.to_csv(PLAYER_OUTPUT, index=False)

    ball_frames = (
        ball_df["frame"].nunique()
        if not ball_df.empty
        else 0
    )

    player_frames = (
        player_df["frame"].nunique()
        if not player_df.empty
        else 0
    )

    print()
    print("=" * 70)
    print("DETECTION COMPLETE")
    print("=" * 70)

    print(f"Frames processed: {processed}")

    print()
    print("BALL")
    print("-" * 70)
    print(f"Ball detections: {len(ball_df)}")
    print(f"Ball frames: {ball_frames}")
    print(
        f"Ball coverage: "
        f"{ball_frames / max(processed, 1) * 100:.2f}%"
    )

    if not ball_df.empty:
        print(
            f"Average ball confidence: "
            f"{ball_df['confidence'].mean():.3f}"
        )

    print()
    print("PLAYER")
    print("-" * 70)
    print(f"Player detections: {len(player_df)}")
    print(f"Player frames: {player_frames}")
    print(
        f"Player frame coverage: "
        f"{player_frames / max(processed, 1) * 100:.2f}%"
    )

    if not player_df.empty:
        print(
            f"Average player confidence: "
            f"{player_df['confidence'].mean():.3f}"
        )

    print()
    print(f"Saved ball detections:   {BALL_OUTPUT}")
    print(f"Saved player detections: {PLAYER_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()