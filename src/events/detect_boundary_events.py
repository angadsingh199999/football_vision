import os
import pandas as pd


# ============================================================
# INPUT / OUTPUT
# ============================================================

TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"
OUTPUT_CSV = "outputs/events/boundary_events.csv"


# ============================================================
# FIELD / VIDEO BOUNDARIES
# ============================================================

# Video is 1280 x 720.
#
# We intentionally use a relatively conservative horizontal
# boundary region. A ball near the camera edge is NOT
# automatically considered a corner.
#
# These are still boundary candidates, not confirmed corners.

LEFT_BOUNDARY = 180
RIGHT_BOUNDARY = 1100


# ============================================================
# DETECTION PARAMETERS
# ============================================================

MIN_SPEED = 5.0

# Maximum time gap between consecutive boundary detections
# belonging to the same boundary sequence.
MAX_GAP = 2.0

# Minimum number of detections required for a normal
# boundary sequence.
MIN_DETECTIONS = 2

# How much the ball must subsequently move back toward
# the field before we consider the boundary interaction
# meaningful.
MIN_RETURN_MOVEMENT = 100.0

# How far inside the field the ball must move after reaching
# the boundary.
#
# LEFT:
#     x should increase
#
# RIGHT:
#     x should decrease
#
# This prevents a single noisy boundary detection from
# becoming a boundary event.
MIN_INSIDE_X = 80.0

# How much time after the boundary peak we inspect.
RETURN_WINDOW = 3.0


# ============================================================
# HELPERS
# ============================================================

def distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return (dx * dx + dy * dy) ** 0.5


def get_boundary(x):

    if x <= LEFT_BOUNDARY:
        return "LEFT"

    if x >= RIGHT_BOUNDARY:
        return "RIGHT"

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(TRAJECTORY_CSV):
        raise FileNotFoundError(
            f"Trajectory file not found: {TRAJECTORY_CSV}"
        )

    df = pd.read_csv(TRAJECTORY_CSV)

    required_columns = {
        "frame",
        "time",
        "center_x",
        "center_y",
        "speed",
        "confidence",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Trajectory CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Create raw boundary detections.
    # --------------------------------------------------------

    boundary_rows = []

    for _, row in df.iterrows():

        x = float(row["center_x"])
        y = float(row["center_y"])
        speed = float(row["speed"])
        time = float(row["time"])

        boundary = get_boundary(x)

        if boundary is None:
            continue

        if speed < MIN_SPEED:
            continue

        boundary_rows.append({
            "frame": int(row["frame"]),
            "time": time,
            "center_x": x,
            "center_y": y,
            "speed": speed,
            "confidence": float(row["confidence"]),
            "boundary": boundary,
        })

    raw = pd.DataFrame(boundary_rows)

    if raw.empty:

        raw.to_csv(
            OUTPUT_CSV,
            index=False
        )

        print("=" * 60)
        print("BOUNDARY EVENTS")
        print("=" * 60)
        print("Raw boundary detections: 0")
        print("Validated boundary events: 0")
        print()
        print(f"Saved: {OUTPUT_CSV}")
        print("=" * 60)

        return

    raw = raw.sort_values(
        "time"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Group consecutive detections on the same boundary.
    # --------------------------------------------------------

    groups = []
    current = []

    for _, row in raw.iterrows():

        if not current:
            current.append(row)
            continue

        previous = current[-1]

        time_gap = (
            float(row["time"])
            - float(previous["time"])
        )

        same_boundary = (
            row["boundary"]
            == previous["boundary"]
        )

        if (
            time_gap <= MAX_GAP
            and same_boundary
        ):
            current.append(row)

        else:
            groups.append(current)
            current = [row]

    if current:
        groups.append(current)

    # --------------------------------------------------------
    # Validate each boundary group.
    # --------------------------------------------------------

    final = []

    for boundary_event_id, rows in enumerate(
        groups,
        start=1
    ):

        group = pd.DataFrame(rows)

        if len(group) < MIN_DETECTIONS:
            continue

        boundary = str(
            group.iloc[0]["boundary"]
        )

        # ----------------------------------------------------
        # Find the point closest to the corresponding
        # horizontal boundary.
        # ----------------------------------------------------

        if boundary == "LEFT":

            peak_idx = group["center_x"].idxmin()

        else:

            peak_idx = group["center_x"].idxmax()

        peak = group.loc[peak_idx]

        peak_time = float(
            peak["time"]
        )

        peak_x = float(
            peak["center_x"]
        )

        peak_y = float(
            peak["center_y"]
        )

        # ----------------------------------------------------
        # Look AFTER the boundary interaction.
        #
        # This is the important correction.
        #
        # We require the ball to actually move back toward
        # the field instead of merely touching the boundary.
        # ----------------------------------------------------

        after = df[
            (
                df["time"] > peak_time
            )
            &
            (
                df["time"]
                <= peak_time + RETURN_WINDOW
            )
        ].copy()

        if after.empty:
            continue

        # ----------------------------------------------------
        # For LEFT boundary:
        #
        # boundary x is small.
        # Ball should subsequently move RIGHT / inward.
        #
        # For RIGHT boundary:
        #
        # boundary x is large.
        # Ball should subsequently move LEFT / inward.
        # ----------------------------------------------------

        if boundary == "LEFT":

            inward = after[
                after["center_x"]
                >= peak_x + MIN_INSIDE_X
            ]

        else:

            inward = after[
                after["center_x"]
                <= peak_x - MIN_INSIDE_X
            ]

        if inward.empty:
            continue

        # Pick the furthest validated point back inside.
        if boundary == "LEFT":

            return_idx = inward["center_x"].idxmax()

        else:

            return_idx = inward["center_x"].idxmin()

        return_row = inward.loc[return_idx]

        return_time = float(
            return_row["time"]
        )

        return_x = float(
            return_row["center_x"]
        )

        return_y = float(
            return_row["center_y"]
        )

        movement_after = distance(
            peak_x,
            peak_y,
            return_x,
            return_y
        )

        # ----------------------------------------------------
        # Calculate how far inside the field the ball moved.
        # ----------------------------------------------------

        if boundary == "LEFT":

            inside_movement = (
                return_x - peak_x
            )

        else:

            inside_movement = (
                peak_x - return_x
            )

        if inside_movement < MIN_INSIDE_X:
            continue

        if movement_after < MIN_RETURN_MOVEMENT:
            continue

        # ----------------------------------------------------
        # Boundary score.
        #
        # This is NOT a corner probability.
        #
        # It measures strength of the boundary interaction.
        # ----------------------------------------------------

        speed_score = min(
            float(peak["speed"]) / 100.0,
            1.0
        )

        movement_score = min(
            movement_after / 500.0,
            1.0
        )

        detection_score = min(
            len(group) / 10.0,
            1.0
        )

        boundary_score = (
            0.30 * speed_score
            + 0.45 * movement_score
            + 0.25 * detection_score
        )

        final.append({
            "boundary_event_id": boundary_event_id,
            "start_time": float(
                group["time"].min()
            ),
            "end_time": float(
                group["time"].max()
            ),
            "peak_time": peak_time,
            "center_x": peak_x,
            "center_y": peak_y,
            "speed": float(
                peak["speed"]
            ),
            "boundary": boundary,
            "detections": len(group),
            "return_time": return_time,
            "return_x": return_x,
            "return_y": return_y,
            "movement_after": movement_after,
            "inside_movement": inside_movement,
            "boundary_score": boundary_score,
        })

    # --------------------------------------------------------
    # Save result.
    # --------------------------------------------------------

    output = pd.DataFrame(final)

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # Print summary.
    # --------------------------------------------------------

    print("=" * 60)
    print("BOUNDARY EVENTS")
    print("=" * 60)

    print(
        f"Raw boundary detections: {len(raw)}"
    )

    print(
        f"Validated boundary events: {len(output)}"
    )

    print()

    if output.empty:

        print(
            "No validated boundary events found."
        )

    else:

        print(
            output.to_string(index=False)
        )

    print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()