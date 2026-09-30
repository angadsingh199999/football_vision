import os
import pandas as pd


# ============================================================
# INPUTS
# ============================================================

CANDIDATE_CSV = (
    "outputs/events/final_corner_candidates.csv"
)

TRACKING_CSV = (
    "outputs/tracking/tracking_data.csv"
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
    "outputs/events/final_corner_events.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

# Player context window around the boundary event.
PLAYER_TIME_WINDOW = 0.5

# Player radius around the boundary ball position.
PLAYER_RADIUS_50 = 50.0
PLAYER_RADIUS_100 = 100.0
PLAYER_RADIUS_150 = 150.0

# Minimum player context.
MIN_PLAYERS_50 = 3
MIN_PLAYERS_100 = 8

# Minimum final corner candidate score.
MIN_CORNER_SCORE = 0.70

# Strong event overlap.
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


def count_unique_players(
    tracking,
    peak_time,
    corner_x,
    corner_y,
    radius,
):

    nearby = tracking[
        (
            tracking["time"]
            >= peak_time - PLAYER_TIME_WINDOW
        )
        &
        (
            tracking["time"]
            <= peak_time + PLAYER_TIME_WINDOW
        )
    ].copy()

    if nearby.empty:
        return 0

    dx = (
        nearby["center_x"]
        - corner_x
    )

    dy = (
        nearby["center_y"]
        - corner_y
    )

    distance = (
        dx * dx
        + dy * dy
    ) ** 0.5

    nearby = nearby[
        distance <= radius
    ]

    if nearby.empty:
        return 0

    # IMPORTANT:
    # Count unique track IDs rather than every detection.
    return int(
        nearby["track_id"]
        .dropna()
        .nunique()
    )


def nearest_player_distance(
    tracking,
    peak_time,
    corner_x,
    corner_y,
):

    nearby = tracking[
        (
            tracking["time"]
            >= peak_time - PLAYER_TIME_WINDOW
        )
        &
        (
            tracking["time"]
            <= peak_time + PLAYER_TIME_WINDOW
        )
    ].copy()

    if nearby.empty:
        return None

    dx = (
        nearby["center_x"]
        - corner_x
    )

    dy = (
        nearby["center_y"]
        - corner_y
    )

    distance = (
        dx * dx
        + dy * dy
    ) ** 0.5

    if distance.empty:
        return None

    return float(
        distance.min()
    )


def has_nearby_event(
    peak_time,
    dataframe,
    time_column,
    tolerance,
):

    if dataframe.empty:
        return False

    for _, row in dataframe.iterrows():

        event_time = float(
            row[time_column]
        )

        if abs(
            peak_time - event_time
        ) <= tolerance:

            return True

    return False


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate files.
    # --------------------------------------------------------

    require_file(CANDIDATE_CSV)
    require_file(TRACKING_CSV)
    require_file(SHOT_CSV)
    require_file(SAVE_CSV)
    require_file(GOAL_CSV)

    candidates = pd.read_csv(
        CANDIDATE_CSV
    )

    tracking = pd.read_csv(
        TRACKING_CSV
    )

    shots = pd.read_csv(
        SHOT_CSV
    )

    saves = pd.read_csv(
        SAVE_CSV
    )

    goals = pd.read_csv(
        GOAL_CSV
    )

    if candidates.empty:

        print("=" * 70)
        print("FINAL CORNER EVENTS")
        print("=" * 70)
        print("No corner candidates found.")
        print()
        print(f"Saved: {OUTPUT_CSV}")
        print("=" * 70)

        pd.DataFrame().to_csv(
            OUTPUT_CSV,
            index=False
        )

        return

    # --------------------------------------------------------
    # Validate tracking columns.
    # --------------------------------------------------------

    required_tracking = {
        "time",
        "track_id",
        "class_name",
        "center_x",
        "center_y",
    }

    missing = (
        required_tracking
        - set(tracking.columns)
    )

    if missing:

        raise ValueError(
            "tracking_data.csv is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Only use players.
    # --------------------------------------------------------

    tracking = tracking[
        tracking["class_name"]
        .astype(str)
        .str.lower()
        == "player"
    ].copy()

    # --------------------------------------------------------
    # Validate candidate columns.
    # --------------------------------------------------------

    required_candidates = {
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
    }

    missing = (
        required_candidates
        - set(candidates.columns)
    )

    if missing:

        raise ValueError(
            "final_corner_candidates.csv is missing "
            "columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # ========================================================
    # ANALYZE CANDIDATES
    # ========================================================

    final = []
    rejected = []

    for _, row in candidates.iterrows():

        corner_id = int(
            row["corner_id"]
        )

        peak_time = float(
            row["peak_time"]
        )

        corner_x = float(
            row["corner_x"]
        )

        corner_y = float(
            row["corner_y"]
        )

        candidate_score = float(
            row["candidate_score"]
        )

        boundary_score = float(
            row["boundary_score"]
        )

        # ----------------------------------------------------
        # PLAYER CONTEXT
        # ----------------------------------------------------

        players_50 = count_unique_players(
            tracking,
            peak_time,
            corner_x,
            corner_y,
            PLAYER_RADIUS_50,
        )

        players_100 = count_unique_players(
            tracking,
            peak_time,
            corner_x,
            corner_y,
            PLAYER_RADIUS_100,
        )

        players_150 = count_unique_players(
            tracking,
            peak_time,
            corner_x,
            corner_y,
            PLAYER_RADIUS_150,
        )

        nearest_player = (
            nearest_player_distance(
                tracking,
                peak_time,
                corner_x,
                corner_y,
            )
        )

        # ----------------------------------------------------
        # PLAYER CONTEXT CHECK
        # ----------------------------------------------------

        if players_50 < MIN_PLAYERS_50:

            rejected.append({
                "corner_id": corner_id,
                "reason": "insufficient_players_within_50",
            })

            continue

        if players_100 < MIN_PLAYERS_100:

            rejected.append({
                "corner_id": corner_id,
                "reason": "insufficient_players_within_100",
            })

            continue

        # ----------------------------------------------------
        # SCORE CHECK
        # ----------------------------------------------------

        if candidate_score < MIN_CORNER_SCORE:

            rejected.append({
                "corner_id": corner_id,
                "reason": "weak_corner_candidate_score",
            })

            continue

        # ----------------------------------------------------
        # STRONGER EVENT CHECK
        # ----------------------------------------------------

        overlaps_goal = has_nearby_event(
            peak_time,
            goals,
            "goal_time",
            GOAL_TOLERANCE,
        )

        overlaps_save = has_nearby_event(
            peak_time,
            saves,
            "shot_time",
            SAVE_TOLERANCE,
        )

        overlaps_shot = has_nearby_event(
            peak_time,
            shots,
            "peak_time",
            SHOT_TOLERANCE,
        )

        if overlaps_goal:

            rejected.append({
                "corner_id": corner_id,
                "reason": "overlaps_goal",
            })

            continue

        if overlaps_save:

            rejected.append({
                "corner_id": corner_id,
                "reason": "overlaps_save",
            })

            continue

        if overlaps_shot:

            rejected.append({
                "corner_id": corner_id,
                "reason": "overlaps_shot",
            })

            continue

        # ----------------------------------------------------
        # FINAL CORNER SCORE
        #
        # Candidate score:
        #   boundary movement + return evidence
        #
        # Player context:
        #   unique players around the location
        #
        # This is still a heuristic score, NOT probability.
        # ----------------------------------------------------

        player_50_score = min(
            players_50 / 8.0,
            1.0
        )

        player_100_score = min(
            players_100 / 15.0,
            1.0
        )

        player_context_score = (
            0.60 * player_50_score
            + 0.40 * player_100_score
        )

        final_corner_score = (
            0.70 * candidate_score
            + 0.30 * player_context_score
        )

        # ----------------------------------------------------
        # Keep the event.
        # ----------------------------------------------------

        final.append({
            "corner_id": corner_id,
            "boundary_event_id": int(
                row["boundary_event_id"]
            ),
            "peak_time": peak_time,
            "boundary": row["boundary"],
            "corner_x": corner_x,
            "corner_y": corner_y,
            "boundary_score": boundary_score,
            "movement_after": float(
                row["movement_after"]
            ),
            "inside_movement": float(
                row["inside_movement"]
            ),
            "detections_after": int(
                row["detections_after"]
            ),
            "boundary_detections": int(
                row["boundary_detections"]
            ),
            "return_score": float(
                row["return_score"]
            ),
            "candidate_score": candidate_score,
            "nearest_player": nearest_player,
            "players_within_50": players_50,
            "players_within_100": players_100,
            "players_within_150": players_150,
            "corner_score": final_corner_score,
        })

    # ========================================================
    # SORT
    # ========================================================

    result = pd.DataFrame(
        final
    )

    if not result.empty:

        result = result[
            result["corner_score"]
            >= MIN_CORNER_SCORE
        ].copy()

        result = result.sort_values(
            "corner_score",
            ascending=False
        ).reset_index(
            drop=True
        )

        result["final_corner_id"] = range(
            1,
            len(result) + 1
        )

        result = result[
            [
                "final_corner_id",
                "corner_id",
                "boundary_event_id",
                "peak_time",
                "boundary",
                "corner_x",
                "corner_y",
                "corner_score",
                "nearest_player",
                "players_within_50",
                "players_within_100",
                "players_within_150",
                "candidate_score",
                "boundary_score",
                "movement_after",
                "inside_movement",
                "detections_after",
                "boundary_detections",
                "return_score",
            ]
        ]

    else:

        result = pd.DataFrame(
            columns=[
                "final_corner_id",
                "corner_id",
                "boundary_event_id",
                "peak_time",
                "boundary",
                "corner_x",
                "corner_y",
                "corner_score",
                "nearest_player",
                "players_within_50",
                "players_within_100",
                "players_within_150",
                "candidate_score",
                "boundary_score",
                "movement_after",
                "inside_movement",
                "detections_after",
                "boundary_detections",
                "return_score",
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
    print("FINAL CORNER EVENTS")
    print("=" * 70)

    print(
        f"Candidates received : {len(candidates)}"
    )

    print(
        f"Final corners       : {len(result)}"
    )

    print(
        f"Rejected candidates : {len(rejected)}"
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

        print("REJECTED CANDIDATES")
        print("-" * 70)

        for item in rejected:

            print(
                f"Corner {item['corner_id']}: "
                f"{item['reason']}"
            )

    print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()