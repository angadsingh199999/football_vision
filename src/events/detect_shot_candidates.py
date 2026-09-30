import pandas as pd
import numpy as np
from pathlib import Path


INPUT_CSV = "outputs/tracking/ball_trajectory.csv"
OUTPUT_CSV = "outputs/events/shot_candidates.csv"


# =========================================================
# CONFIGURATION
# =========================================================

MIN_CONFIDENCE = 0.20

# Candidate movement threshold.
MIN_SPEED = 35.0

# Minimum displacement during candidate movement.
MIN_HORIZONTAL_DISTANCE = 50.0
MIN_TOTAL_DISTANCE = 70.0

# A football shot is normally predominantly horizontal
# in this broadcast camera.
MIN_HORIZONTAL_RATIO = 0.60

# Look around a speed peak.
BACKWARD_POINTS = 5
FORWARD_POINTS = 8

# Maximum candidate temporal span.
MAX_WINDOW_TIME = 1.20

# Minimum number of trajectory observations in the window.
MIN_SUPPORT = 2

# Nearby peaks represent the same movement.
DEDUP_TIME = 1.75

# Existing video dimensions.
VIDEO_CENTER_X = 640.0

# Goal-mouth approximation.
LEFT_GOAL_X = 200.0
RIGHT_GOAL_X = 1080.0


# =========================================================
# HELPERS
# =========================================================

def safe_float(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):

        return default


def distance(x1, y1, x2, y2):

    return float(
        np.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )
    )


def calculate_window_features(window):

    if len(window) < 2:
        return None

    start = window.iloc[0]
    end = window.iloc[-1]

    start_x = safe_float(
        start["center_x"]
    )

    start_y = safe_float(
        start["center_y"]
    )

    end_x = safe_float(
        end["center_x"]
    )

    end_y = safe_float(
        end["center_y"]
    )

    horizontal_distance = abs(
        end_x - start_x
    )

    total_distance = distance(
        start_x,
        start_y,
        end_x,
        end_y
    )

    if total_distance <= 0:
        return None

    horizontal_ratio = (
        horizontal_distance /
        total_distance
    )

    duration = (
        safe_float(end["time"])
        -
        safe_float(start["time"])
    )

    if duration <= 0:
        return None

    return {
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "horizontal_distance":
            horizontal_distance,
        "total_distance":
            total_distance,
        "horizontal_ratio":
            horizontal_ratio,
        "duration":
            duration,
    }


def determine_direction(start_x, end_x):

    if end_x > start_x:
        return "RIGHT"

    if end_x < start_x:
        return "LEFT"

    return "UNKNOWN"


def direction_toward_goal(
    start_x,
    end_x
):

    midpoint = (
        start_x +
        end_x
    ) / 2.0

    direction = determine_direction(
        start_x,
        end_x
    )

    if midpoint < VIDEO_CENTER_X:

        return (
            direction == "RIGHT"
        )

    return (
        direction == "LEFT"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    Path(OUTPUT_CSV).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        INPUT_CSV
    )

    print("=" * 70)
    print("SHOT CANDIDATE DETECTION")
    print("=" * 70)

    print(
        f"Trajectory rows loaded: {len(df)}"
    )

    required = {
        "segment_id",
        "frame",
        "time",
        "confidence",
        "center_x",
        "center_y",
        "speed",
    }

    missing = (
        required -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing columns: "
            +
            str(sorted(missing))
        )

    numeric = [
        "segment_id",
        "frame",
        "time",
        "confidence",
        "center_x",
        "center_y",
        "speed",
    ]

    for column in numeric:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric
    )

    df = df[
        df["confidence"] >=
        MIN_CONFIDENCE
    ].copy()

    df = df.sort_values(
        ["segment_id", "frame"]
    ).reset_index(
        drop=True
    )

    candidates = []

    # =====================================================
    # PROCESS EACH TRACKLET
    # =====================================================

    for segment_id, segment in df.groupby(
        "segment_id",
        sort=True
    ):

        segment = (
            segment
            .sort_values("frame")
            .reset_index(drop=True)
        )

        if len(segment) < MIN_SUPPORT:
            continue

        # -------------------------------------------------
        # Every sufficiently fast point becomes a possible
        # movement peak.
        # -------------------------------------------------

        for peak_idx in range(
            len(segment)
        ):

            peak = segment.iloc[
                peak_idx
            ]

            peak_speed = safe_float(
                peak["speed"]
            )

            if peak_speed < MIN_SPEED:
                continue

            start_idx = max(
                0,
                peak_idx -
                BACKWARD_POINTS
            )

            end_idx = min(
                len(segment),
                peak_idx +
                FORWARD_POINTS +
                1
            )

            window = segment.iloc[
                start_idx:end_idx
            ].copy()

            if len(window) < MIN_SUPPORT:
                continue

            features = calculate_window_features(
                window
            )

            if features is None:
                continue

            if (
                features["duration"]
                >
                MAX_WINDOW_TIME
            ):
                continue

            if (
                features["horizontal_distance"]
                <
                MIN_HORIZONTAL_DISTANCE
            ):
                continue

            if (
                features["total_distance"]
                <
                MIN_TOTAL_DISTANCE
            ):
                continue

            if (
                features["horizontal_ratio"]
                <
                MIN_HORIZONTAL_RATIO
            ):
                continue

            # -------------------------------------------------
            # Direction.
            # -------------------------------------------------

            direction = determine_direction(
                features["start_x"],
                features["end_x"]
            )

            if direction == "UNKNOWN":
                continue

            # -------------------------------------------------
            # We don't reject based on goal direction here.
            #
            # Why?
            # A pass can look like a shot and a shot can be
            # directed diagonally. This stage should preserve
            # candidates for later semantic filtering.
            # -------------------------------------------------

            peak_time = safe_float(
                peak["time"]
            )

            confidence = safe_float(
                peak["confidence"]
            )

            # -------------------------------------------------
            # Candidate confidence.
            # -------------------------------------------------

            speed_score = min(
                peak_speed / 80.0,
                1.0
            )

            distance_score = min(
                features[
                    "total_distance"
                ] / 300.0,
                1.0
            )

            horizontal_score = min(
                features[
                    "horizontal_ratio"
                ],
                1.0
            )

            candidate_confidence = (
                0.45 * speed_score
                +
                0.30 * distance_score
                +
                0.15 * horizontal_score
                +
                0.10 * confidence
            )

            candidates.append(
                {
                    "segment_id":
                        int(segment_id),

                    "shot_time":
                        peak_time,

                    "start_time":
                        safe_float(
                            window["time"].min()
                        ),

                    "end_time":
                        safe_float(
                            window["time"].max()
                        ),

                    "peak_frame":
                        int(
                            peak["frame"]
                        ),

                    "peak_speed":
                        peak_speed,

                    "start_x":
                        features["start_x"],

                    "start_y":
                        features["start_y"],

                    "end_x":
                        features["end_x"],

                    "end_y":
                        features["end_y"],

                    "horizontal_distance":
                        features[
                            "horizontal_distance"
                        ],

                    "total_distance":
                        features[
                            "total_distance"
                        ],

                    "horizontal_ratio":
                        features[
                            "horizontal_ratio"
                        ],

                    "direction":
                        direction,

                    "confidence":
                        confidence,

                    "candidate_confidence":
                        candidate_confidence,

                    "support_points":
                        len(window),
                }
            )

    # =====================================================
    # SORT
    # =====================================================

    candidates_df = pd.DataFrame(
        candidates
    )

    if candidates_df.empty:

        candidates_df = pd.DataFrame(
            columns=[
                "segment_id",
                "shot_time",
                "start_time",
                "end_time",
                "peak_frame",
                "peak_speed",
                "start_x",
                "start_y",
                "end_x",
                "end_y",
                "horizontal_distance",
                "total_distance",
                "horizontal_ratio",
                "direction",
                "confidence",
                "candidate_confidence",
                "support_points",
            ]
        )

    else:

        candidates_df = (
            candidates_df
            .sort_values(
                [
                    "shot_time",
                    "candidate_confidence",
                ],
                ascending=[
                    True,
                    False,
                ]
            )
            .reset_index(drop=True)
        )

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    final_candidates = []

    for _, candidate in candidates_df.iterrows():

        if not final_candidates:

            final_candidates.append(
                candidate.to_dict()
            )

            continue

        previous = final_candidates[-1]

        time_difference = abs(
            candidate["shot_time"]
            -
            previous["shot_time"]
        )

        if time_difference <= DEDUP_TIME:

            # Keep the stronger candidate.
            if (
                candidate[
                    "candidate_confidence"
                ]
                >
                previous[
                    "candidate_confidence"
                ]
            ):

                final_candidates[-1] = (
                    candidate.to_dict()
                )

        else:

            final_candidates.append(
                candidate.to_dict()
            )

    final_df = pd.DataFrame(
        final_candidates
    )

    if not final_df.empty:

        final_df.insert(
            0,
            "shot_id",
            range(
                1,
                len(final_df) + 1
            )
        )

        final_df = final_df.sort_values(
            "shot_time"
        ).reset_index(
            drop=True
        )

    final_df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("=" * 70)
    print("SHOT DETECTION COMPLETE")
    print("=" * 70)

    print(
        f"Shot candidates: "
        f"{len(final_df)}"
    )

    if not final_df.empty:

        print()

        display_columns = [
            "shot_id",
            "shot_time",
            "start_time",
            "end_time",
            "peak_speed",
            "total_distance",
            "horizontal_ratio",
            "direction",
            "candidate_confidence",
            "support_points",
        ]

        print(
            final_df[
                display_columns
            ].to_string(index=False)
        )

    print()
    print(
        f"Saved: {OUTPUT_CSV}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()