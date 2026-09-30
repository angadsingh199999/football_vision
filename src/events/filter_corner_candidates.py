import os
import pandas as pd


# ============================================================
# INPUTS
# ============================================================

BOUNDARY_ANALYSIS_CSV = (
    "outputs/events/analyzed_boundary_events.csv"
)

SHOT_CSV = (
    "outputs/events/final_shot_events.csv"
)

SAVE_CSV = (
    "outputs/events/final_save_events.csv"
)

GOAL_CSV = (
    "outputs/events/goal_events.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_CSV = (
    "outputs/events/final_corner_candidates.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

# Minimum boundary evidence.
MIN_BOUNDARY_SCORE = 0.50

# Minimum movement back into the pitch.
MIN_INSIDE_MOVEMENT = 100.0

# Minimum total movement after boundary.
MIN_MOVEMENT_AFTER = 150.0

# Minimum trajectory detections after boundary.
MIN_DETECTIONS_AFTER = 8

# Minimum return score.
MIN_RETURN_SCORE = 0.55

# ------------------------------------------------------------
# Event overlap windows.
# ------------------------------------------------------------

SHOT_TOLERANCE = 2.0
SAVE_TOLERANCE = 2.0
GOAL_TOLERANCE = 3.0


# ============================================================
# HELPERS
# ============================================================

def require_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def time_close(time_a, time_b, tolerance):

    return (
        abs(
            float(time_a)
            - float(time_b)
        )
        <= tolerance
    )


# ============================================================
# LOAD STRONGER EVENTS
# ============================================================

def load_stronger_events():

    require_file(SHOT_CSV)
    require_file(SAVE_CSV)
    require_file(GOAL_CSV)

    shots = pd.read_csv(
        SHOT_CSV
    )

    saves = pd.read_csv(
        SAVE_CSV
    )

    goals = pd.read_csv(
        GOAL_CSV
    )

    return shots, saves, goals


# ============================================================
# CHECK WHETHER BOUNDARY EVENT OVERLAPS A STRONGER EVENT
# ============================================================

def find_stronger_overlap(
    peak_time,
    shots,
    saves,
    goals,
):

    # --------------------------------------------------------
    # GOAL
    # --------------------------------------------------------

    for _, row in goals.iterrows():

        goal_time = float(
            row["goal_time"]
        )

        if time_close(
            peak_time,
            goal_time,
            GOAL_TOLERANCE,
        ):

            return (
                "GOAL",
                goal_time,
            )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    for _, row in saves.iterrows():

        save_time = float(
            row["shot_time"]
        )

        if time_close(
            peak_time,
            save_time,
            SAVE_TOLERANCE,
        ):

            return (
                "SAVE",
                save_time,
            )

    # --------------------------------------------------------
    # SHOT
    # --------------------------------------------------------

    for _, row in shots.iterrows():

        shot_time = float(
            row["peak_time"]
        )

        if time_close(
            peak_time,
            shot_time,
            SHOT_TOLERANCE,
        ):

            return (
                "SHOT",
                shot_time,
            )

    return (
        None,
        None,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    require_file(
        BOUNDARY_ANALYSIS_CSV
    )

    boundaries = pd.read_csv(
        BOUNDARY_ANALYSIS_CSV
    )

    if boundaries.empty:

        print(
            "No analyzed boundary events found."
        )

        return

    shots, saves, goals = (
        load_stronger_events()
    )

    candidates = []

    rejected = []

    # ========================================================
    # PROCESS EACH BOUNDARY EVENT
    # ========================================================

    for _, row in boundaries.iterrows():

        event_id = int(
            row["boundary_event_id"]
        )

        peak_time = float(
            row["peak_time"]
        )

        boundary = str(
            row["boundary"]
        )

        boundary_score = float(
            row["boundary_score"]
        )

        inside_movement = float(
            row["inside_movement"]
        )

        movement_after = float(
            row["movement_after"]
        )

        detections_after = int(
            row["detections_after"]
        )

        return_score = float(
            row["return_score"]
        )

        valid_return = bool(
            row["valid_return"]
        )

        boundary_detections = int(
            row["boundary_detections"]
        )

        # ----------------------------------------------------
        # RULE 1: Valid boundary return.
        # ----------------------------------------------------

        if not valid_return:

            rejected.append({
                "boundary_event_id": event_id,
                "reason": "invalid_boundary_return"
            })

            continue

        # ----------------------------------------------------
        # RULE 2: Boundary strength.
        # ----------------------------------------------------

        if boundary_score < MIN_BOUNDARY_SCORE:

            rejected.append({
                "boundary_event_id": event_id,
                "reason": "weak_boundary_score"
            })

            continue

        # ----------------------------------------------------
        # RULE 3: Inward movement.
        # ----------------------------------------------------

        if inside_movement < MIN_INSIDE_MOVEMENT:

            rejected.append({
                "boundary_event_id": event_id,
                "reason": "insufficient_inside_movement"
            })

            continue

        # ----------------------------------------------------
        # RULE 4: Total movement.
        # ----------------------------------------------------

        if movement_after < MIN_MOVEMENT_AFTER:

            rejected.append({
                "boundary_event_id": event_id,
                "reason": "insufficient_return_movement"
            })

            continue

        # ----------------------------------------------------
        # RULE 5: Post-event detections.
        # ----------------------------------------------------

        if detections_after < MIN_DETECTIONS_AFTER:

            rejected.append({
                "boundary_event_id": event_id,
                "reason": "insufficient_post_boundary_detections"
            })

            continue

        # ----------------------------------------------------
        # RULE 6: Return score.
        # ----------------------------------------------------

        if return_score < MIN_RETURN_SCORE:

            rejected.append({
                "boundary_event_id": event_id,
                "reason": "weak_return_score"
            })

            continue

        # ----------------------------------------------------
        # RULE 7: Need enough detections AT the boundary.
        # ----------------------------------------------------

        if boundary_detections < 2:

            rejected.append({
                "boundary_event_id": event_id,
                "reason": "insufficient_boundary_detections"
            })

            continue

        # ----------------------------------------------------
        # RULE 8: Stronger event overlap.
        #
        # A boundary interaction occurring immediately around
        # a shot/save/goal should not become an independent
        # corner highlight.
        # ----------------------------------------------------

        stronger_type, stronger_time = (
            find_stronger_overlap(
                peak_time,
                shots,
                saves,
                goals,
            )
        )

        if stronger_type is not None:

            rejected.append({
                "boundary_event_id": event_id,
                "reason": (
                    "overlaps_"
                    + stronger_type.lower()
                ),
                "stronger_event_time": stronger_time,
            })

            continue

        # ----------------------------------------------------
        # Candidate score.
        #
        # This is a CORNER CANDIDATE score.
        # It is NOT a probability.
        # ----------------------------------------------------

        boundary_strength = min(
            boundary_score,
            1.0
        )

        movement_strength = min(
            movement_after / 700.0,
            1.0
        )

        inside_strength = min(
            inside_movement / 700.0,
            1.0
        )

        detection_strength = min(
            detections_after / 30.0,
            1.0
        )

        return_strength = min(
            return_score,
            1.0
        )

        candidate_score = (
            0.20 * boundary_strength
            + 0.25 * movement_strength
            + 0.20 * inside_strength
            + 0.15 * detection_strength
            + 0.20 * return_strength
        )

        candidates.append({
            "boundary_event_id": event_id,
            "peak_time": peak_time,
            "boundary": boundary,
            "corner_x": float(
                row["start_x"]
            ),
            "corner_y": float(
                row["start_y"]
            ),
            "boundary_score": boundary_score,
            "movement_after": movement_after,
            "inside_movement": inside_movement,
            "detections_after": detections_after,
            "boundary_detections": boundary_detections,
            "return_score": return_score,
            "candidate_score": candidate_score,
        })

    # ========================================================
    # SORT
    # ========================================================

    result = pd.DataFrame(
        candidates
    )

    if not result.empty:

        result = result.sort_values(
            "candidate_score",
            ascending=False
        ).reset_index(
            drop=True
        )

        result["corner_id"] = range(
            1,
            len(result) + 1
        )

        result = result[
            [
                "corner_id",
                "boundary_event_id",
                "peak_time",
                "boundary",
                "corner_x",
                "corner_y",
                "boundary_score",
                "movement_after",
                "inside_movement",
                "detections_after",
                "boundary_detections",
                "return_score",
                "candidate_score",
            ]
        ]

    else:

        result = pd.DataFrame(
            columns=[
                "corner_id",
                "boundary_event_id",
                "peak_time",
                "boundary",
                "corner_x",
                "corner_y",
                "boundary_score",
                "movement_after",
                "inside_movement",
                "detections_after",
                "boundary_detections",
                "return_score",
                "candidate_score",
            ]
        )

    # ========================================================
    # SAVE
    # ========================================================

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
    print("CORNER CANDIDATES")
    print("=" * 70)

    print(
        f"Boundary events received : {len(boundaries)}"
    )

    print(
        f"Corner candidates        : {len(result)}"
    )

    print(
        f"Rejected boundary events : {len(rejected)}"
    )

    print()

    if not result.empty:

        print(
            result.to_string(
                index=False
            )
        )

    print()

    if rejected:

        print("REJECTED EVENTS")
        print("-" * 70)

        for item in rejected:

            print(
                f"Boundary {item['boundary_event_id']}: "
                f"{item['reason']}"
            )

    print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()