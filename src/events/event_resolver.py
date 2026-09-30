import os
import pandas as pd


EVENT_DIR = "outputs/events"

GOAL_FILE = os.path.join(
    EVENT_DIR,
    "goal_events.csv"
)

SHOT_FILE = os.path.join(
    EVENT_DIR,
    "final_shot_events.csv"
)

SAVE_FILE = os.path.join(
    EVENT_DIR,
    "final_save_events.csv"
)

OUTPUT_FILE = os.path.join(
    EVENT_DIR,
    "resolved_event_index.csv"
)


# ------------------------------------------------------------
# FINAL EVENT TYPES
# ------------------------------------------------------------

ALLOWED_TYPES = {
    "GOAL",
    "SAVE",
    "SHOT",
}


# ------------------------------------------------------------
# CLIP WINDOWS
# ------------------------------------------------------------

GOAL_BEFORE = 3.0
GOAL_AFTER = 3.0

SAVE_BEFORE = 2.0
SAVE_AFTER = 2.0

SHOT_BEFORE = 2.0
SHOT_AFTER = 2.0


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


def require_file(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


def load_goals():

    require_file(GOAL_FILE)

    df = pd.read_csv(
        GOAL_FILE
    )

    events = []

    for index, row in df.iterrows():

        peak = safe_float(
            row.get("goal_time", 0)
        )

        events.append(
            {
                "event_type": "GOAL",
                "event_id": index + 1,
                "peak_time": peak,
                "start_time": peak - GOAL_BEFORE,
                "end_time": peak + GOAL_AFTER,
                "confidence": 1.0,
                "supporting_events": "",
                "resolution_reason": "scoreboard_confirmed",
            }
        )

    return events


def load_shots():

    require_file(SHOT_FILE)

    df = pd.read_csv(
        SHOT_FILE
    )

    events = []

    for index, row in df.iterrows():

        peak = safe_float(
            row.get("peak_time", 0)
        )

        confidence = safe_float(
            row.get(
                "peak_confidence",
                0
            )
        )

        events.append(
            {
                "event_type": "SHOT",
                "event_id": int(
                    row.get(
                        "event_id",
                        index + 1
                    )
                ),
                "peak_time": peak,
                "start_time": peak - SHOT_BEFORE,
                "end_time": peak + SHOT_AFTER,
                "confidence": confidence,
                "supporting_events": "",
                "resolution_reason": "initial_shot",
            }
        )

    return events


def load_saves():

    require_file(SAVE_FILE)

    df = pd.read_csv(
        SAVE_FILE
    )

    events = []

    for index, row in df.iterrows():

        peak = safe_float(
            row.get("shot_time", 0)
        )

        confidence = safe_float(
            row.get(
                "save_confidence",
                0.9
            )
        )

        shot_id = row.get(
            "shot_id",
            ""
        )

        events.append(
            {
                "event_type": "SAVE",
                "event_id": int(
                    index + 1
                ),
                "peak_time": peak,
                "start_time": peak - SAVE_BEFORE,
                "end_time": peak + SAVE_AFTER,
                "confidence": confidence,
                "supporting_events": (
                    f"SHOT:{shot_id}"
                    if str(shot_id) != ""
                    else ""
                ),
                "resolution_reason": (
                    "save_absorbs_shot"
                ),
            }
        )

    return events


def overlaps(event_a, event_b):

    return (
        event_a["start_time"]
        <= event_b["end_time"]
        and
        event_b["start_time"]
        <= event_a["end_time"]
    )


def resolve_events():

    goals = load_goals()
    saves = load_saves()
    shots = load_shots()

    print("=" * 70)
    print("EVENT RESOLUTION")
    print("=" * 70)

    print(
        f"Goals : {len(goals)}"
    )

    print(
        f"Saves : {len(saves)}"
    )

    print(
        f"Shots : {len(shots)}"
    )

    print()

    # --------------------------------------------------------
    # GOALS HAVE HIGHEST PRIORITY
    # --------------------------------------------------------

    resolved = []

    for goal in goals:
        resolved.append(goal)

    # --------------------------------------------------------
    # SAVES ABSORB SHOTS
    # --------------------------------------------------------

    for save in saves:

        # If the save itself is too close to a goal,
        # goal wins.
        near_goal = False

        for goal in goals:

            if abs(
                save["peak_time"]
                - goal["peak_time"]
            ) <= 4.0:

                near_goal = True
                break

        if not near_goal:
            resolved.append(save)

    # --------------------------------------------------------
    # SHOTS SURVIVE ONLY IF THEY ARE NOT PART OF A SAVE
    # OR GOAL.
    # --------------------------------------------------------

    for shot in shots:

        absorbed = False

        # Goal absorbs nearby shot.
        for goal in goals:

            if abs(
                shot["peak_time"]
                - goal["peak_time"]
            ) <= 4.0:

                absorbed = True

                break

        if absorbed:
            continue

        # Save absorbs nearby shot.
        for save in saves:

            if abs(
                shot["peak_time"]
                - save["peak_time"]
            ) <= 3.0:

                absorbed = True

                break

        if absorbed:
            continue

        resolved.append(shot)

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    resolved.sort(
        key=lambda event:
        event["peak_time"]
    )

    # --------------------------------------------------------
    # FINAL SAFETY PASS
    #
    # No two final events may have overlapping clip
    # intervals. Higher-priority events win.
    # --------------------------------------------------------

    priority = {
        "GOAL": 3,
        "SAVE": 2,
        "SHOT": 1,
    }

    final_events = []

    for event in resolved:

        if event["event_type"] not in ALLOWED_TYPES:
            continue

        conflict = None

        for existing in final_events:

            if overlaps(
                event,
                existing
            ):

                conflict = existing
                break

        if conflict is None:

            final_events.append(
                event
            )

        else:

            current_priority = priority[
                event["event_type"]
            ]

            existing_priority = priority[
                conflict["event_type"]
            ]

            if current_priority > existing_priority:

                final_events.remove(
                    conflict
                )

                final_events.append(
                    event
                )

    final_events.sort(
        key=lambda event:
        event["peak_time"]
    )

    # --------------------------------------------------------
    # CREATE FINAL CSV
    # --------------------------------------------------------

    output_rows = []

    for final_id, event in enumerate(
        final_events,
        start=1
    ):

        output_rows.append(
            {
                "final_event_id": final_id,
                "event_type": event[
                    "event_type"
                ],
                "event_id": event[
                    "event_id"
                ],
                "start_time": round(
                    max(
                        0.0,
                        event["start_time"]
                    ),
                    2
                ),
                "end_time": round(
                    event["end_time"],
                    2
                ),
                "peak_time": round(
                    event["peak_time"],
                    2
                ),
                "confidence": round(
                    event["confidence"],
                    4
                ),
                "supporting_events": event[
                    "supporting_events"
                ],
                "resolution_reason": event[
                    "resolution_reason"
                ],
            }
        )

    result = pd.DataFrame(
        output_rows
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Final events: {len(result)}"
    )

    print()

    if not result.empty:

        print(
            result.to_string(
                index=False
            )
        )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    resolve_events()