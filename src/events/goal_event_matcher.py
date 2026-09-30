import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCORE_JSON = Path(
    "outputs/goal_candidates.json"
)

MOVEMENT_CSV = Path(
    "outputs/events/movement_events.csv"
)

OUTPUT_CSV = Path(
    "outputs/events/goal_events.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# OCR usually reports the score after the actual goal.
#
# We therefore search BACKWARD from the OCR score-change
# timestamp and use the last strong ball movement before it.

MAX_LOOKBACK = 20.0
MAX_LOOKAHEAD = 1.0
MAX_MOVEMENT_START_AFTER_SCORE = 0.25
MAX_LOCALIZATION_GAP = 10.0
SCOREBOARD_TO_GOAL_OFFSET = 3.0

# Ignore extremely tiny movement events.
MIN_MOVEMENT_SPEED = 30.0

# Minimum movement duration.
MIN_MOVEMENT_DURATION = 0.0

# Goal clip window.
GOAL_BEFORE = 5.0
GOAL_AFTER = 5.0


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# LOAD SCORE CHANGES
# ============================================================

def load_score_changes():

    if not SCORE_JSON.exists():

        raise FileNotFoundError(
            f"Score candidate file not found: {SCORE_JSON}"
        )

    with open(
        SCORE_JSON,
        "r"
    ) as f:

        return json.load(f)


# ============================================================
# LOAD MOVEMENT EVENTS
# ============================================================

def load_movement_events():

    if not MOVEMENT_CSV.exists():

        raise FileNotFoundError(
            f"Movement event file not found: {MOVEMENT_CSV}"
        )

    df = pd.read_csv(
        MOVEMENT_CSV
    )

    required = {
        "event_id",
        "start_time",
        "end_time",
        "peak_time",
        "peak_speed",
        "max_confidence",
        "detections",
        "duration",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "movement_events.csv is missing columns: "
            + str(sorted(missing))
        )

    numeric_columns = [
        "start_time",
        "end_time",
        "peak_time",
        "peak_speed",
        "max_confidence",
        "detections",
        "duration",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "start_time",
            "end_time",
            "peak_time",
            "peak_speed",
        ]
    )

    return df.sort_values(
        "end_time"
    ).reset_index(
        drop=True
    )


# ============================================================
# FIND ACTUAL GOAL MOVEMENT
# ============================================================

def find_goal_movement(
    score_time,
    movements
):

    # Scoreboards usually update after the goal. A movement episode can still
    # be in progress when the score becomes visible, so compare its peak time
    # (the strongest localized point) rather than requiring its end to precede
    # the OCR timestamp.
    candidates = movements[
        (
            (movements["peak_time"] <= score_time)
            | (
                (movements["start_time"] <= score_time + MAX_MOVEMENT_START_AFTER_SCORE)
                & (movements["peak_time"] <= score_time + MAX_LOOKAHEAD)
            )
        )
        & (movements["peak_time"] >= score_time - MAX_LOOKBACK)
        &
        (
            movements["peak_speed"]
            >= MIN_MOVEMENT_SPEED
        )
        &
        (
            movements["duration"]
            >= MIN_MOVEMENT_DURATION
        )
    ].copy()

    if candidates.empty:

        return None

    # The strongest clue is the movement closest to the
    # scoreboard-confirmed score change.
    candidates["gap"] = (score_time - candidates["peak_time"]).abs()

    candidates = candidates.sort_values(
        [
            "gap",
            "peak_speed",
        ],
        ascending=[
            True,
            False,
        ]
    )

    return candidates.iloc[0]


# ============================================================
# CREATE GOAL EVENTS
# ============================================================

def main():

    print("=" * 70)
    print("GOAL EVENT MATCHING - SCOREBOARD + BALL MOVEMENT")
    print("=" * 70)

    score_events = load_score_changes()
    movements = load_movement_events()

    print(
        f"Score changes detected : {len(score_events)}"
    )

    print(
        f"Movement events loaded : {len(movements)}"
    )

    print()

    goal_events = []

    for index, goal in enumerate(
        score_events,
        start=1
    ):

        score_time = safe_float(
            goal["video_time"]
        )

        previous_home = int(
            goal["previous_home_score"]
        )

        previous_away = int(
            goal["previous_away_score"]
        )

        new_home = int(
            goal["new_home_score"]
        )

        new_away = int(
            goal["new_away_score"]
        )

        # ----------------------------------------------------
        # Determine scoring team
        # ----------------------------------------------------

        if new_home > previous_home:

            scoring_team = goal["home_team"]

        elif new_away > previous_away:

            scoring_team = goal["away_team"]

        else:

            scoring_team = "UNKNOWN"

        # ----------------------------------------------------
        # Find actual scoring movement
        # ----------------------------------------------------

        movement = find_goal_movement(
            score_time,
            movements
        )

        if movement is not None:
            movement_gap = score_time - safe_float(movement["peak_time"])
            if movement_gap > MAX_LOCALIZATION_GAP:
                movement = None

        if movement is None:
            # Score changes are the reliable event signal. Ball tracking can
            # miss a shot, so keep the confirmed goal and estimate its time
            # shortly before the scoreboard update instead of dropping it or
            # attaching it to an unrelated older movement.
            actual_goal_time = max(
                0.0, score_time - SCOREBOARD_TO_GOAL_OFFSET,
            )
            movement_fields = {
                "movement_event_id": None,
                "movement_start": None,
                "movement_end": None,
                "movement_peak_time": None,
                "movement_peak_speed": None,
            }
            confidence = 0.35
            event_source = "scoreboard_confirmed_time_estimate"
            print(
                f"GOAL {index}: scoreboard confirmed at {score_time:.2f}s; "
                f"no nearby movement, estimated goal time "
                f"{actual_goal_time:.2f}s"
            )
        else:
            actual_goal_time = safe_float(movement["peak_time"])
            gap = score_time - actual_goal_time
            movement_fields = {
                "movement_event_id": int(movement["event_id"]),
                "movement_start": safe_float(movement["start_time"]),
                "movement_end": safe_float(movement["end_time"]),
                "movement_peak_time": safe_float(movement["peak_time"]),
                "movement_peak_speed": safe_float(movement["peak_speed"]),
            }
            confidence = round(
                min(
                    0.90,
                    max(
                        0.35,
                        0.40
                        + 0.35 * safe_float(movement["max_confidence"])
                        + 0.15 * max(0.0, 1.0 - gap / MAX_LOOKBACK),
                    ),
                ),
                3,
            )
            event_source = "scoreboard_confirmed_ball_movement_localized"

        clip_start = max(
            0.0,
            actual_goal_time - GOAL_BEFORE
        )

        clip_end = (
            actual_goal_time
            + GOAL_AFTER
        )

        gap = (
            score_time
            - actual_goal_time
        )

        print(f"GOAL {index}")
        print(f"  OCR score change : {score_time:.2f}s")
        if movement is not None:
            print(
                f"  Ball movement    : {movement['start_time']:.2f}s -> "
                f"{movement['end_time']:.2f}s"
            )
        else:
            print("  Ball movement    : not detected; using score-change estimate")
        print(f"  Actual goal time : {actual_goal_time:.2f}s")
        print(f"  OCR delay        : {gap:.2f}s")
        print()

        goal_events.append({

            "goal_id": index,

            "goal_time": actual_goal_time,

            "clock": goal["clock"],

            "home_team": goal["home_team"],

            "away_team": goal["away_team"],

            "scoring_team": scoring_team,

            "previous_score": (
                f"{previous_home}-{previous_away}"
            ),

            "new_score": (
                f"{new_home}-{new_away}"
            ),

            "clip_start": clip_start,

            "clip_end": clip_end,

            "confidence": confidence,

            "score_change_time": score_time,

            **movement_fields,

            "event_source": event_source,

        })

    # ========================================================
    # SAVE
    # ========================================================

    result = pd.DataFrame(
        goal_events,
        columns=[
            "goal_id", "goal_time", "clock", "home_team", "away_team",
            "scoring_team", "previous_score", "new_score", "clip_start",
            "clip_end", "confidence", "score_change_time",
            "movement_event_id", "movement_start", "movement_end",
            "movement_peak_time", "movement_peak_speed", "event_source",
        ],
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 70)
    print("GOAL EVENTS")
    print("=" * 70)

    if result.empty:

        print(
            "No goal events generated."
        )

    else:

        print(
            result.to_string(
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
