from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("outputs/tracking/ball_detections.csv")
OUTPUT = Path("outputs/tracking/ball_trajectory.csv")


# ============================================================
# TRACKING SETTINGS
# ============================================================

MIN_CONFIDENCE = 0.15

# Maximum frame gap allowed while extending a tracklet.
MAX_FRAME_GAP = 5

# Maximum allowed spatial jump for a single frame.
BASE_MAX_JUMP = 90.0

# Additional tolerance for gaps.
GAP_JUMP_ALLOWANCE = 35.0

# Tracklet must have at least this many detections.
MIN_TRACK_LENGTH = 3


def distance(x1, y1, x2, y2):
    return float(np.hypot(x2 - x1, y2 - y1))


def main():

    print("=" * 70)
    print("BALL TRAJECTORY - CONTINUOUS BALL TRACKLETS")
    print("=" * 70)

    if not INPUT.exists():
        raise FileNotFoundError(f"Missing: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "frame",
        "time",
        "confidence",
        "center_x",
        "center_y",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in ball detections: {sorted(missing)}"
        )

    df = df[
        df["confidence"] >= MIN_CONFIDENCE
    ].copy()

    df = df.sort_values(
        ["frame", "confidence"],
        ascending=[True, False]
    ).reset_index(drop=True)

    print(f"Raw detections after confidence filter: {len(df)}")

    if df.empty:
        pd.DataFrame(
            columns=[
                "segment_id",
                "frame",
                "time",
                "confidence",
                "center_x",
                "center_y",
                "dx",
                "dy",
                "speed",
                "acceleration",
            ]
        ).to_csv(OUTPUT, index=False)

        print("No ball detections.")
        return

    # --------------------------------------------------------
    # Build greedy continuous tracklets.
    #
    # Each detection can belong to at most one tracklet.
    # We prefer candidates that are close to the predicted
    # position of an existing track.
    # --------------------------------------------------------

    detections = df.to_dict("records")

    tracks = []

    for det in detections:

        frame = int(det["frame"])
        x = float(det["center_x"])
        y = float(det["center_y"])

        best_track = None
        best_cost = float("inf")

        for track in tracks:

            last = track[-1]

            last_frame = int(last["frame"])
            gap = frame - last_frame

            if gap <= 0:
                continue

            if gap > MAX_FRAME_GAP:
                continue

            # Estimate velocity from last two points if possible.
            if len(track) >= 2:

                p1 = track[-2]
                p2 = track[-1]

                dt = max(
                    int(p2["frame"]) - int(p1["frame"]),
                    1,
                )

                vx = (
                    float(p2["center_x"]) -
                    float(p1["center_x"])
                ) / dt

                vy = (
                    float(p2["center_y"]) -
                    float(p1["center_y"])
                ) / dt

            else:

                vx = 0.0
                vy = 0.0

            predicted_x = (
                float(last["center_x"]) +
                vx * gap
            )

            predicted_y = (
                float(last["center_y"]) +
                vy * gap
            )

            d_pred = distance(
                predicted_x,
                predicted_y,
                x,
                y,
            )

            d_last = distance(
                float(last["center_x"]),
                float(last["center_y"]),
                x,
                y,
            )

            max_jump = (
                BASE_MAX_JUMP +
                GAP_JUMP_ALLOWANCE * (gap - 1)
            )

            if d_pred > max_jump and d_last > max_jump:
                continue

            # Prediction is preferred over raw last-point distance.
            cost = min(d_pred, d_last)

            # Higher confidence slightly improves the score.
            cost -= float(det["confidence"]) * 10.0

            if cost < best_cost:
                best_cost = cost
                best_track = track

        if best_track is None:

            tracks.append([det])

        else:

            best_track.append(det)

    # --------------------------------------------------------
    # Keep useful tracklets.
    # --------------------------------------------------------

    useful_tracks = [
        track
        for track in tracks
        if len(track) >= MIN_TRACK_LENGTH
    ]

    print(f"Initial tracklets: {len(tracks)}")
    print(f"Useful tracklets: {len(useful_tracks)}")

    output_rows = []

    segment_id = 0

    for track in useful_tracks:

        segment_id += 1

        track = sorted(
            track,
            key=lambda r: int(r["frame"])
        )

        previous = None
        previous_speed = 0.0

        for det in track:

            frame = int(det["frame"])
            time_sec = float(det["time"])

            x = float(det["center_x"])
            y = float(det["center_y"])

            if previous is None:

                dx = 0.0
                dy = 0.0
                speed = 0.0
                acceleration = 0.0

            else:

                frame_gap = max(
                    frame - previous["frame"],
                    1
                )

                dx = (
                    x - previous["x"]
                ) / frame_gap

                dy = (
                    y - previous["y"]
                ) / frame_gap

                speed = float(
                    np.hypot(dx, dy)
                )

                acceleration = (
                    speed -
                    previous_speed
                )

            output_rows.append(
                {
                    "segment_id": segment_id,
                    "frame": frame,
                    "time": time_sec,
                    "confidence": float(det["confidence"]),
                    "center_x": x,
                    "center_y": y,
                    "dx": dx,
                    "dy": dy,
                    "speed": speed,
                    "acceleration": acceleration,
                }
            )

            previous = {
                "frame": frame,
                "x": x,
                "y": y,
            }

            previous_speed = speed

    output = pd.DataFrame(
        output_rows,
        columns=[
            "segment_id",
            "frame",
            "time",
            "confidence",
            "center_x",
            "center_y",
            "dx",
            "dy",
            "speed",
            "acceleration",
        ],
    )

    output = output.sort_values(
        ["segment_id", "frame"]
    ).reset_index(drop=True)

    output.to_csv(OUTPUT, index=False)

    print()
    print("=" * 70)
    print("TRAJECTORY COMPLETE")
    print("=" * 70)

    print(f"Trajectory rows: {len(output)}")
    print(
        f"Trajectory segments: "
        f"{output['segment_id'].nunique() if not output.empty else 0}"
    )

    if not output.empty:
        print(
            f"Time: "
            f"{output['time'].min():.2f} -> "
            f"{output['time'].max():.2f} seconds"
        )

        print(
            f"Maximum speed: "
            f"{output['speed'].max():.2f} pixels/frame"
        )

        print(
            f"Average speed: "
            f"{output['speed'].mean():.2f} pixels/frame"
        )

    print()
    print(f"Saved: {OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()