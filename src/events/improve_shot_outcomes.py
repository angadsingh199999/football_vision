import os
import pandas as pd


SHOT_CSV = "outputs/events/final_shot_events.csv"
GOAL_CSV = "outputs/events/goal_events.csv"
SAVE_CSV = "outputs/events/final_save_events.csv"

OUTPUT_CSV = "outputs/events/improved_shot_outcomes.csv"

GOAL_TOLERANCE = 1.0
SAVE_TOLERANCE = 1.0


def load_times(path, possible_columns):

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
    print("IMPROVING SHOT OUTCOMES")
    print("=" * 50)

    # --------------------------------------------------
    # LOAD SHOTS
    # --------------------------------------------------

    shots = pd.read_csv(SHOT_CSV)

    if shots.empty:
        print("No shot events found.")
        return

    if "peak_time" not in shots.columns:
        raise ValueError(
            "final_shot_events.csv must contain "
            "'peak_time'."
        )

    # --------------------------------------------------
    # LOAD GOALS
    # --------------------------------------------------

    goal_times = load_times(
        GOAL_CSV,
        [
            "goal_time",
            "video_time",
            "peak_time",
            "time"
        ]
    )

    # --------------------------------------------------
    # LOAD SAVES
    # --------------------------------------------------

    save_times = load_times(
        SAVE_CSV,
        [
            "shot_time",
            "peak_time",
            "time"
        ]
    )

    print(f"Shots loaded: {len(shots)}")
    print(f"Goal events loaded: {len(goal_times)}")
    print(f"Save events loaded: {len(save_times)}")

    # --------------------------------------------------
    # CLASSIFY EACH SHOT
    # --------------------------------------------------

    outcomes = []

    for _, shot in shots.iterrows():

        shot_time = float(
            shot["peak_time"]
        )

        outcome = "NO_GOAL"

        # ----------------------------------------------
        # CHECK GOAL
        # ----------------------------------------------

        goal_match = any(
            abs(shot_time - goal_time)
            <= GOAL_TOLERANCE
            for goal_time in goal_times
        )

        if goal_match:

            outcome = "GOAL"

        else:

            # ------------------------------------------
            # CHECK SAVE
            # ------------------------------------------

            save_match = any(
                abs(shot_time - save_time)
                <= SAVE_TOLERANCE
                for save_time in save_times
            )

            if save_match:
                outcome = "SAVE"

        outcomes.append(
            outcome
        )

    # --------------------------------------------------
    # CREATE OUTPUT
    # --------------------------------------------------

    result = shots.copy()

    result["outcome"] = outcomes

    result = result.sort_values(
        "peak_time"
    ).reset_index(drop=True)

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
    print("SHOT OUTCOMES")
    print()

    print(
        result[
            [
                "shot_id",
                "peak_time",
                "direction",
                "peak_speed",
                "peak_confidence",
                "detections",
                "outcome"
            ]
        ].to_string(index=False)
    )

    print()
    print("Outcome counts:")
    print(
        result["outcome"].value_counts()
    )

    print()
    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()