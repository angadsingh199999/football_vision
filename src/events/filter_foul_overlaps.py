import os
import pandas as pd


FOUL_CSV = "outputs/events/final_foul_candidates.csv"
OUTPUT_CSV = "outputs/events/filtered_foul_candidates.csv"


EVENT_FILES = {
    "GOAL": (
        "outputs/events/goal_events.csv",
        ["peak_time", "video_time", "time", "goal_time"]
    ),

    "SHOT": (
        "outputs/events/final_shot_events.csv",
        ["peak_time", "shot_time", "time"]
    ),

    "SAVE": (
        "outputs/events/final_save_events.csv",
        ["shot_time", "peak_time", "time"]
    ),

    "ATTACK": (
        "outputs/events/final_dangerous_attacks.csv",
        ["peak_time", "time"]
    ),

    "CORNER": (
        "outputs/events/final_corner_events.csv",
        ["peak_time", "time"]
    ),
}


OVERLAP_TOLERANCE = 3.0


def get_event_times(path, possible_columns):

    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        return []

    df = pd.read_csv(path)

    for column in possible_columns:

        if column in df.columns:

            values = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()

            return values.tolist()

    print(
        f"Warning: no usable time column found in {path}"
    )

    return []


def main():

    print("=" * 50)
    print("FILTERING FOUL EVENT OVERLAPS")
    print("=" * 50)

    foul_df = pd.read_csv(FOUL_CSV)

    if foul_df.empty:
        print("No foul candidates found.")
        return

    if "peak_time" not in foul_df.columns:
        raise ValueError(
            "final_foul_candidates.csv must contain "
            "'peak_time'."
        )

    # --------------------------------------------------
    # LOAD EXISTING EVENT TIMES
    # --------------------------------------------------

    event_times = {}

    for event_type, (path, columns) in EVENT_FILES.items():

        times = get_event_times(
            path,
            columns
        )

        event_times[event_type] = times

        print(
            f"{event_type}: {len(times)} events loaded"
        )

    # --------------------------------------------------
    # FILTER FOUL CANDIDATES
    # --------------------------------------------------

    kept = []
    removed = []

    for _, row in foul_df.iterrows():

        foul_time = float(
            row["peak_time"]
        )

        overlapping_events = []

        for event_type, times in event_times.items():

            for event_time in times:

                if abs(
                    foul_time - float(event_time)
                ) <= OVERLAP_TOLERANCE:

                    overlapping_events.append(
                        event_type
                    )

                    break

        if overlapping_events:

            removed.append({
                "foul_candidate_id": int(
                    row["foul_candidate_id"]
                ),
                "peak_time": foul_time,
                "reason": ", ".join(
                    overlapping_events
                )
            })

        else:

            kept.append(row)

    # --------------------------------------------------
    # CREATE RESULT
    # --------------------------------------------------

    result = pd.DataFrame(kept)

    if not result.empty:

        result = result.sort_values(
            "foul_score",
            ascending=False
        ).reset_index(drop=True)

        result["final_foul_id"] = range(
            1,
            len(result) + 1
        )

        columns = [
            "final_foul_id",
            "foul_candidate_id",
            "peak_time",
            "start_time",
            "end_time",
            "ball_x",
            "ball_y",
            "nearby_players",
            "nearest_player",
            "foul_score",
            "detections"
        ]

        result = result[columns]

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print()

    print(
        f"Original foul candidates: "
        f"{len(foul_df)}"
    )

    print(
        f"Removed because of known events: "
        f"{len(removed)}"
    )

    print(
        f"Remaining foul candidates: "
        f"{len(result)}"
    )

    print()

    if not result.empty:

        print(
            result.to_string(index=False)
        )

    else:

        print(
            "No foul-only candidates remain."
        )

    print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()