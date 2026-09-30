import pandas as pd
import numpy as np
from pathlib import Path


INPUT_CSV = "outputs/tracking/ball_detections.csv"
OUTPUT_CSV = "outputs/tracking/ball_trajectory.csv"

# ---------------------------------------------------------
# Detection filtering
# ---------------------------------------------------------

MIN_CONFIDENCE = 0.20

# ---------------------------------------------------------
# Association
# ---------------------------------------------------------

# We only allow very small temporal gaps.
# A missing frame means we do not know where the ball is.
MAX_FRAME_GAP = 2

# Maximum displacement for one frame.
MAX_JUMP_PER_FRAME = 100.0

# Maximum distance from previous accepted position.
MAX_ASSOCIATION_DISTANCE = 120.0

# Confidence bonus when choosing between nearby detections.
CONFIDENCE_WEIGHT = 30.0

# ---------------------------------------------------------
# Segment requirements
# ---------------------------------------------------------

# Minimum number of observations for a useful segment.
MIN_SEGMENT_POINTS = 3


def safe_float(value, default=0.0):

    try:
        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


def euclidean_distance(
    x1,
    y1,
    x2,
    y2
):

    return float(
        np.sqrt(
            (x2 - x1) ** 2
            +
            (y2 - y1) ** 2
        )
    )


def choose_best_detection(
    frame_df,
    previous
):

    candidates = []

    previous_x = previous["center_x"]
    previous_y = previous["center_y"]

    frame_gap = (
        int(frame_df["frame"].iloc[0])
        -
        int(previous["frame"])
    )

    max_distance = (
        MAX_ASSOCIATION_DISTANCE
        +
        MAX_JUMP_PER_FRAME
        *
        max(0, frame_gap - 1)
    )

    for _, row in frame_df.iterrows():

        x = safe_float(
            row["center_x"]
        )

        y = safe_float(
            row["center_y"]
        )

        conf = safe_float(
            row["confidence"]
        )

        d = euclidean_distance(
            previous_x,
            previous_y,
            x,
            y
        )

        if d > max_distance:
            continue

        # Lower score is better.
        #
        # Distance is the dominant signal.
        # Confidence is only a small tie-breaker.
        score = (
            d
            -
            CONFIDENCE_WEIGHT * conf
        )

        candidates.append(
            (
                score,
                d,
                row
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0]
    )

    return candidates[0][2]


def calculate_speed(
    previous,
    current
):

    frame_gap = (
        current["frame"]
        -
        previous["frame"]
    )

    if frame_gap <= 0:
        return 0.0

    d = euclidean_distance(
        previous["center_x"],
        previous["center_y"],
        current["center_x"],
        current["center_y"]
    )

    return d / frame_gap


def finalize_segment(
    segment,
    segments
):

    if len(segment) >= MIN_SEGMENT_POINTS:

        segments.append(
            segment
        )


def main():

    Path(
        OUTPUT_CSV
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        INPUT_CSV
    )

    print("=" * 70)
    print("STRICT BALL TRAJECTORY")
    print("=" * 70)

    print(
        f"Raw detections: {len(df)}"
    )

    required = {
        "frame",
        "time",
        "confidence",
        "center_x",
        "center_y"
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    # -----------------------------------------------------
    # Numeric cleanup
    # -----------------------------------------------------

    for column in [
        "frame",
        "time",
        "confidence",
        "center_x",
        "center_y"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "frame",
            "time",
            "confidence",
            "center_x",
            "center_y"
        ]
    )

    df["frame"] = (
        df["frame"]
        .astype(int)
    )

    # -----------------------------------------------------
    # Confidence filter
    # -----------------------------------------------------

    df = df[
        df["confidence"] >= MIN_CONFIDENCE
    ].copy()

    print(
        f"After confidence filtering: "
        f"{len(df)}"
    )

    if df.empty:

        print(
            "No detections remain."
        )

        pd.DataFrame().to_csv(
            OUTPUT_CSV,
            index=False
        )

        return

    # -----------------------------------------------------
    # Sort detections
    # -----------------------------------------------------

    df = df.sort_values(
        [
            "frame",
            "confidence"
        ],
        ascending=[
            True,
            False
        ]
    )

    # -----------------------------------------------------
    # Process frame by frame.
    #
    # IMPORTANT:
    # We keep ALL detections until association.
    # We do NOT simply choose highest confidence.
    # -----------------------------------------------------

    grouped = df.groupby(
        "frame",
        sort=True
    )

    segments = []

    current_segment = []

    previous = None

    rejected = 0

    new_segments = 0

    for frame_number, frame_df in grouped:

        frame_number = int(
            frame_number
        )

        frame_df = frame_df.copy()

        # -------------------------------------------------
        # Start first segment
        # -------------------------------------------------

        if previous is None:

            selected = frame_df.iloc[0]

            current_segment = [{
                "frame": frame_number,
                "time": safe_float(
                    selected["time"]
                ),
                "confidence": safe_float(
                    selected["confidence"]
                ),
                "center_x": safe_float(
                    selected["center_x"]
                ),
                "center_y": safe_float(
                    selected["center_y"]
                )
            }]

            previous = current_segment[-1]

            new_segments += 1

            continue

        # -------------------------------------------------
        # Temporal gap
        # -------------------------------------------------

        frame_gap = (
            frame_number
            -
            previous["frame"]
        )

        # If more than two frames are missing,
        # terminate the current trajectory.
        if frame_gap > MAX_FRAME_GAP:

            finalize_segment(
                current_segment,
                segments
            )

            current_segment = []

            previous = None

            # Start a NEW segment with the strongest
            # detection in this frame.
            selected = frame_df.iloc[0]

            current_segment = [{
                "frame": frame_number,
                "time": safe_float(
                    selected["time"]
                ),
                "confidence": safe_float(
                    selected["confidence"]
                ),
                "center_x": safe_float(
                    selected["center_x"]
                ),
                "center_y": safe_float(
                    selected["center_y"]
                )
            }]

            previous = current_segment[-1]

            new_segments += 1

            continue

        # -------------------------------------------------
        # Find spatially compatible detection.
        # -------------------------------------------------

        selected = choose_best_detection(
            frame_df,
            previous
        )

        # -------------------------------------------------
        # No compatible detection.
        # -------------------------------------------------

        if selected is None:

            rejected += len(
                frame_df
            )

            finalize_segment(
                current_segment,
                segments
            )

            current_segment = []

            previous = None

            new_segments += 1

            # Do NOT immediately reconnect to another
            # arbitrary detection.
            continue

        # -------------------------------------------------
        # Calculate movement.
        # -------------------------------------------------

        current = {
            "frame": frame_number,
            "time": safe_float(
                selected["time"]
            ),
            "confidence": safe_float(
                selected["confidence"]
            ),
            "center_x": safe_float(
                selected["center_x"]
            ),
            "center_y": safe_float(
                selected["center_y"]
            )
        }

        movement = euclidean_distance(
            previous["center_x"],
            previous["center_y"],
            current["center_x"],
            current["center_y"]
        )

        speed = movement / frame_gap

        # -------------------------------------------------
        # Hard physical limit.
        # -------------------------------------------------

        if speed > MAX_JUMP_PER_FRAME:

            rejected += 1

            finalize_segment(
                current_segment,
                segments
            )

            current_segment = []

            previous = None

            new_segments += 1

            continue

        # -------------------------------------------------
        # Accept detection.
        # -------------------------------------------------

        current_segment.append(
            current
        )

        previous = current

    # -----------------------------------------------------
    # Final segment
    # -----------------------------------------------------

    finalize_segment(
        current_segment,
        segments
    )

    # -----------------------------------------------------
    # Flatten segments
    # -----------------------------------------------------

    rows = []

    for segment_id, segment in enumerate(
        segments,
        start=1
    ):

        for item in segment:

            item = item.copy()

            item["segment_id"] = (
                segment_id
            )

            rows.append(
                item
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        print(
            "No coherent trajectory segments found."
        )

        result.to_csv(
            OUTPUT_CSV,
            index=False
        )

        return

    # -----------------------------------------------------
    # Calculate movement inside each segment only.
    # -----------------------------------------------------

    result = result.sort_values(
        [
            "segment_id",
            "frame"
        ]
    ).reset_index(
        drop=True
    )

    result["frame_gap"] = (
        result.groupby(
            "segment_id"
        )["frame"]
        .diff()
    )

    result["distance"] = np.sqrt(
        result.groupby(
            "segment_id"
        )["center_x"]
        .diff() ** 2
        +
        result.groupby(
            "segment_id"
        )["center_y"]
        .diff() ** 2
    )

    result["speed"] = (
        result["distance"]
        /
        result["frame_gap"]
    )

    result["distance"] = (
        result["distance"]
        .fillna(0)
    )

    result["speed"] = (
        result["speed"]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    # -----------------------------------------------------
    # Final safety check.
    # -----------------------------------------------------

    result.loc[
        result["speed"] > MAX_JUMP_PER_FRAME,
        "speed"
    ] = 0

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("STRICT TRAJECTORY COMPLETE")
    print("=" * 70)

    print(
        f"Trajectory rows: "
        f"{len(result)}"
    )

    print(
        f"Trajectory segments: "
        f"{result['segment_id'].nunique()}"
    )

    print(
        f"Rejected detections: "
        f"{rejected}"
    )

    print(
        f"Time coverage: "
        f"{result['time'].min():.2f} → "
        f"{result['time'].max():.2f}s"
    )

    print(
        f"Maximum speed: "
        f"{result['speed'].max():.2f} px/frame"
    )

    print(
        f"Average speed: "
        f"{result['speed'].mean():.2f} px/frame"
    )

    print()
    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()