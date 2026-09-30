import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = Path(
    "outputs/scoreboard_readings.json"
)

OUTPUT_PATH = Path(
    "outputs/goal_candidates.json"
)
CSV_OUTPUT_PATH = Path("outputs/score_changes.csv")


# ============================================================
# SETTINGS
# ============================================================

# A new score must remain visible for this many
# consecutive valid OCR readings.
MIN_CONFIRMATIONS = 2
MAX_CONFIRMATION_GAP_SECONDS = 4.0


# ============================================================
# LOAD SCOREBOARD READINGS
# ============================================================

print("=" * 70)
print("SCORE CHANGE DETECTION")
print("=" * 70)

if not INPUT_PATH.exists():

    raise FileNotFoundError(
        f"Scoreboard readings not found: {INPUT_PATH}"
    )


with open(
    INPUT_PATH,
    "r"
) as file:

    readings = json.load(file)


if not isinstance(readings, list):

    raise ValueError(
        "scoreboard_readings.json must contain a JSON list."
    )


print(
    f"Total scoreboard readings: "
    f"{len(readings)}"
)


# ============================================================
# HELPER
# ============================================================

def get_score(reading):

    if not isinstance(
        reading,
        dict
    ):
        return None

    home = reading.get(
        "home_score"
    )

    away = reading.get(
        "away_score"
    )

    # bool is technically an int in Python,
    # so explicitly reject it.
    if isinstance(home, bool):
        return None

    if isinstance(away, bool):
        return None

    if not isinstance(home, int):
        return None

    if not isinstance(away, int):
        return None

    # Sanity check.
    if home < 0 or away < 0:
        return None

    if home > 20 or away > 20:
        return None

    return (
        home,
        away
    )


# ============================================================
# FIND VALID READINGS
# ============================================================

valid_readings = []

for reading in readings:

    score = get_score(
        reading
    )

    if score is not None:

        valid_readings.append(
            (
                reading,
                score
            )
        )


print(
    f"Valid score readings: "
    f"{len(valid_readings)}"
)


# ============================================================
# NO VALID OCR
# ============================================================

if not valid_readings:

    print()
    print(
        "WARNING: No valid scoreboard "
        "scores were detected."
    )

    print(
        "Goal detection cannot be resolved "
        "from scoreboard changes."
    )

    print(
        "Creating empty goal_candidates.json "
        "so the pipeline can continue."
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_PATH,
        "w"
    ) as file:

        json.dump(
            [],
            file,
            indent=4
        )

    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=[
        "video_time", "clock", "home_team", "away_team",
        "previous_home_score", "previous_away_score",
        "new_home_score", "new_away_score", "change_type",
    ]).to_csv(CSV_OUTPUT_PATH, index=False)


    print()
    print("=" * 70)
    print("SCORE CHANGE DETECTION COMPLETE")
    print("Goal candidates: 0")
    print(
        f"Saved to: {OUTPUT_PATH}"
    )
    print(f"Saved score-change table to: {CSV_OUTPUT_PATH}")
    print("=" * 70)

    raise SystemExit(0)


# ============================================================
# FIND FIRST CONFIRMED SCORE
# ============================================================

current_score = None
baseline_score = None
baseline_readings = []


# ============================================================
# DETECT SCORE CHANGES
# ============================================================

goal_candidates = []
score_changes = []

pending_score = None
pending_readings = []


for reading in readings:

    score = get_score(
        reading
    )


    # --------------------------------------------------------
    # OCR did not provide a valid score.
    #
    # IMPORTANT:
    # Do not reset pending candidates just because
    # one OCR frame failed.
    # --------------------------------------------------------

    if score is None:

        continue

    # Establish the match's starting score only after repeated OCR readings.
    # This prevents one bad first frame from shifting the whole score history.
    if current_score is None:
        if score != baseline_score:
            baseline_score = score
            baseline_readings = [reading]
        else:
            previous_time = baseline_readings[-1].get("video_time")
            current_time = reading.get("video_time")
            try:
                if (
                    previous_time is not None
                    and current_time is not None
                    and float(current_time) - float(previous_time)
                    > MAX_CONFIRMATION_GAP_SECONDS
                ):
                    baseline_readings = []
            except (TypeError, ValueError):
                baseline_readings = []
            baseline_readings.append(reading)

        if len(baseline_readings) >= MIN_CONFIRMATIONS:
            current_score = baseline_score
            pending_score = None
            pending_readings = []
        continue


    # --------------------------------------------------------
    # Same as confirmed score
    # --------------------------------------------------------

    if score == current_score:

        pending_score = None
        pending_readings = []

        continue

    home_delta = score[0] - current_score[0]
    away_delta = score[1] - current_score[1]
    is_goal_step = (
        (home_delta == 1 and away_delta == 0)
        or (home_delta == 0 and away_delta == 1)
    )
    is_single_score_rollback = (
        (home_delta == -1 and away_delta == 0)
        or (home_delta == 0 and away_delta == -1)
    )
    if not (is_goal_step or is_single_score_rollback):
        # Large jumps usually come from non-score text (such as jersey
        # numbers) or a partial OCR read. Do not let them reset the baseline.
        pending_score = None
        pending_readings = []
        continue


    # --------------------------------------------------------
    # New candidate score
    # --------------------------------------------------------

    if score != pending_score:

        pending_score = score

        pending_readings = [
            reading
        ]

    else:
        previous_time = pending_readings[-1].get("video_time")
        current_time = reading.get("video_time")
        try:
            if (
                previous_time is not None
                and current_time is not None
                and float(current_time) - float(previous_time)
                > MAX_CONFIRMATION_GAP_SECONDS
            ):
                pending_readings = []
        except (TypeError, ValueError):
            pending_readings = []
        pending_readings.append(reading)


    # --------------------------------------------------------
    # Confirm score change
    # --------------------------------------------------------

    if len(pending_readings) >= MIN_CONFIRMATIONS:

        first_reading = (
            pending_readings[0]
        )

        old_home, old_away = (
            current_score
        )

        new_home, new_away = (
            pending_score
        )


        is_goal_score_step = (
            (new_home - old_home == 1 and new_away == old_away)
            or (new_away - old_away == 1 and new_home == old_home)
        )

        if is_goal_score_step:
            details_reading = next(
                (
                    item for item in pending_readings
                    if item.get("home_team") or item.get("away_team")
                ),
                first_reading,
            )
            clock = next(
                (item.get("clock") for item in pending_readings if item.get("clock")),
                first_reading.get("clock"),
            )

            goal_candidate = {
                "video_time": first_reading.get("video_time"),
                "clock": clock,
                "home_team": details_reading.get("home_team"),
                "away_team": details_reading.get("away_team"),
                "previous_home_score": old_home,
                "previous_away_score": old_away,
                "new_home_score": new_home,
                "new_away_score": new_away,
                "change_type": "goal",
            }
            goal_candidates.append(goal_candidate)
            score_changes.append(goal_candidate)
            print(
                f"GOAL CANDIDATE | {first_reading.get('video_time', 0):.1f}s | "
                f"{old_home}-{old_away} -> {new_home}-{new_away}"
            )
        else:
            # A repeated scoreboard correction (for example a VAR-overturned
            # goal) changes the baseline, but is not itself reported as a goal.
            print(
                f"SCOREBOARD CORRECTION | "
                f"{first_reading.get('video_time', 0):.1f}s | "
                f"{old_home}-{old_away} -> {new_home}-{new_away}"
            )
            score_changes.append({
                "video_time": first_reading.get("video_time"),
                "clock": next(
                    (item.get("clock") for item in pending_readings if item.get("clock")),
                    first_reading.get("clock"),
                ),
                "home_team": next(
                    (item.get("home_team") for item in pending_readings if item.get("home_team")),
                    first_reading.get("home_team"),
                ),
                "away_team": next(
                    (item.get("away_team") for item in pending_readings if item.get("away_team")),
                    first_reading.get("away_team"),
                ),
                "previous_home_score": old_home,
                "previous_away_score": old_away,
                "new_home_score": new_home,
                "new_away_score": new_away,
                "change_type": "scoreboard_correction",
            })


        # New score is confirmed.
        current_score = pending_score

        pending_score = None
        pending_readings = []


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w"
) as file:

    json.dump(
        goal_candidates,
        file,
        indent=4
    )

pd.DataFrame(score_changes, columns=[
    "video_time", "clock", "home_team", "away_team",
    "previous_home_score", "previous_away_score",
    "new_home_score", "new_away_score", "change_type",
]).to_csv(CSV_OUTPUT_PATH, index=False)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SCORE CHANGE DETECTION COMPLETE")
print("=" * 70)
print(
    f"Goal candidates: "
    f"{len(goal_candidates)}"
)
print(
    f"Saved to: "
    f"{OUTPUT_PATH}"
)
print(f"Saved score-change table to: {CSV_OUTPUT_PATH}")
print("=" * 70)
