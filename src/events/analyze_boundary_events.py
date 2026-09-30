import os
import math
import pandas as pd


# ============================================================
# INPUT / OUTPUT
# ============================================================

BOUNDARY_CSV = "outputs/events/boundary_events.csv"
TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"

OUTPUT_CSV = "outputs/events/analyzed_boundary_events.csv"


# ============================================================
# PARAMETERS
# ============================================================

# Look at the ball for several seconds after the boundary
# interaction.
AFTER_WINDOW = 3.0

# Minimum inward movement along the X axis.
MIN_INSIDE_MOVEMENT = 80.0

# Minimum total movement after reaching the boundary.
MIN_MOVEMENT_AFTER = 100.0

# Minimum number of trajectory detections after the event.
MIN_DETECTIONS_AFTER = 5


# ============================================================
# HELPERS
# ============================================================

def euclidean_distance(x1, y1, x2, y2):

    dx = x2 - x1
    dy = y2 - y1

    return math.sqrt(
        dx ** 2 + dy ** 2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(BOUNDARY_CSV):
        raise FileNotFoundError(
            f"Boundary file not found: {BOUNDARY_CSV}"
        )

    if not os.path.exists(TRAJECTORY_CSV):
        raise FileNotFoundError(
            f"Trajectory file not found: {TRAJECTORY_CSV}"
        )

    boundaries = pd.read_csv(
        BOUNDARY_CSV
    )

    trajectory = pd.read_csv(
        TRAJECTORY_CSV
    )

    # --------------------------------------------------------
    # Validate input columns.
    # --------------------------------------------------------

    boundary_required = {
        "boundary_event_id",
        "start_time",
        "end_time",
        "peak_time",
        "center_x",
        "center_y",
        "speed",
        "boundary",
        "detections",
        "return_time",
        "return_x",
        "return_y",
        "movement_after",
        "inside_movement",
        "boundary_score",
    }

    missing_boundary = (
        boundary_required
        - set(boundaries.columns)
    )

    if missing_boundary:
        raise ValueError(
            "boundary_events.csv is missing columns: "
            + ", ".join(
                sorted(missing_boundary)
            )
        )

    trajectory_required = {
        "time",
        "center_x",
        "center_y",
        "speed",
        "confidence",
    }

    missing_trajectory = (
        trajectory_required
        - set(trajectory.columns)
    )

    if missing_trajectory:
        raise ValueError(
            "ball_trajectory.csv is missing columns: "
            + ", ".join(
                sorted(missing_trajectory)
            )
        )

    trajectory = trajectory.sort_values(
        "time"
    ).reset_index(drop=True)

    results = []

    # ========================================================
    # ANALYZE EACH BOUNDARY EVENT
    # ========================================================

    for _, event in boundaries.iterrows():

        event_id = int(
            event["boundary_event_id"]
        )

        boundary = str(
            event["boundary"]
        )

        peak_time = float(
            event["peak_time"]
        )

        start_x = float(
            event["center_x"]
        )

        start_y = float(
            event["center_y"]
        )

        boundary_speed = float(
            event["speed"]
        )

        boundary_detections = int(
            event["detections"]
        )

        boundary_score = float(
            event["boundary_score"]
        )

        # ----------------------------------------------------
        # Get trajectory after boundary interaction.
        # ----------------------------------------------------

        after = trajectory[
            (trajectory["time"] > peak_time)
            &
            (
                trajectory["time"]
                <= peak_time + AFTER_WINDOW
            )
        ].copy()

        if after.empty:
            continue

        # ----------------------------------------------------
        # Determine the furthest inward point.
        #
        # LEFT:
        #     larger X = further inside
        #
        # RIGHT:
        #     smaller X = further inside
        # ----------------------------------------------------

        if boundary == "LEFT":

            inward_idx = after[
                "center_x"
            ].idxmax()

        else:

            inward_idx = after[
                "center_x"
            ].idxmin()

        inward = after.loc[
            inward_idx
        ]

        inward_time = float(
            inward["time"]
        )

        inward_x = float(
            inward["center_x"]
        )

        inward_y = float(
            inward["center_y"]
        )

        # ----------------------------------------------------
        # Calculate X movement toward the pitch.
        # ----------------------------------------------------

        if boundary == "LEFT":

            inside_movement = (
                inward_x - start_x
            )

        else:

            inside_movement = (
                start_x - inward_x
            )

        # ----------------------------------------------------
        # Total movement from boundary point to furthest
        # inward point.
        # ----------------------------------------------------

        movement_after = euclidean_distance(
            start_x,
            start_y,
            inward_x,
            inward_y
        )

        # ----------------------------------------------------
        # Determine whether ball actually moved inward.
        # ----------------------------------------------------

        moved_inside = (
            inside_movement
            >= MIN_INSIDE_MOVEMENT
        )

        # ----------------------------------------------------
        # Number of detections after the event.
        # ----------------------------------------------------

        detections_after = len(
            after
        )

        # ----------------------------------------------------
        # Calculate how quickly the ball moved back inside.
        # ----------------------------------------------------

        return_duration = (
            inward_time
            - peak_time
        )

        if return_duration > 0:

            return_speed = (
                movement_after
                / return_duration
            )

        else:

            return_speed = 0.0

        # ----------------------------------------------------
        # Ball confidence after boundary interaction.
        # ----------------------------------------------------

        mean_confidence_after = float(
            after["confidence"].mean()
        )

        max_speed_after = float(
            after["speed"].max()
        )

        # ----------------------------------------------------
        # Calculate a boundary-return score.
        #
        # IMPORTANT:
        # This is still NOT a "corner probability".
        #
        # It measures how strongly the ball returned into
        # the field after reaching the boundary.
        # ----------------------------------------------------

        movement_score = min(
            movement_after / 700.0,
            1.0
        )

        inside_score = min(
            inside_movement / 600.0,
            1.0
        )

        detection_score = min(
            detections_after / 20.0,
            1.0
        )

        confidence_score = min(
            mean_confidence_after,
            1.0
        )

        return_score = (
            0.35 * movement_score
            + 0.30 * inside_score
            + 0.20 * detection_score
            + 0.15 * confidence_score
        )

        # ----------------------------------------------------
        # Final validation.
        # ----------------------------------------------------

        valid_return = (
            moved_inside
            and
            movement_after
            >= MIN_MOVEMENT_AFTER
            and
            detections_after
            >= MIN_DETECTIONS_AFTER
        )

        results.append({
            "boundary_event_id": event_id,
            "boundary": boundary,
            "peak_time": peak_time,

            "start_x": start_x,
            "start_y": start_y,

            "boundary_speed": boundary_speed,
            "boundary_detections": boundary_detections,
            "boundary_score": boundary_score,

            "return_time": inward_time,
            "end_x": inward_x,
            "end_y": inward_y,

            "movement_after": round(
                movement_after,
                2
            ),

            "inside_movement": round(
                inside_movement,
                2
            ),

            "return_duration": round(
                return_duration,
                3
            ),

            "return_speed": round(
                return_speed,
                2
            ),

            "moved_inside": bool(
                moved_inside
            ),

            "detections_after": detections_after,

            "mean_confidence_after": round(
                mean_confidence_after,
                4
            ),

            "max_speed_after": round(
                max_speed_after,
                2
            ),

            "return_score": round(
                return_score,
                4
            ),

            "valid_return": bool(
                valid_return
            ),
        })

    # ========================================================
    # SAVE
    # ========================================================

    result = pd.DataFrame(
        results
    )

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("=" * 70)
    print("BOUNDARY EVENT ANALYSIS")
    print("=" * 70)

    print(
        f"Boundary events received : {len(boundaries)}"
    )

    print(
        f"Events analyzed          : {len(result)}"
    )

    if not result.empty:

        print()

        print(
            result[
                [
                    "boundary_event_id",
                    "boundary",
                    "peak_time",
                    "start_x",
                    "start_y",
                    "end_x",
                    "end_y",
                    "movement_after",
                    "inside_movement",
                    "detections_after",
                    "return_score",
                    "valid_return",
                ]
            ].to_string(
                index=False
            )
        )

    print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()