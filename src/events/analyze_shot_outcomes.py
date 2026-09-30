import pandas as pd


SHOTS_CSV = "outputs/events/final_shot_events.csv"
GOALS_CSV = "outputs/events/goal_events.csv"

OUTPUT_CSV = "outputs/events/shot_outcomes.csv"

GOAL_PROXIMITY_TIME = 5.0


def main():

    shots = pd.read_csv(SHOTS_CSV)
    goals = pd.read_csv(GOALS_CSV)

    results = []

    for _, shot in shots.iterrows():

        shot_time = float(shot["peak_time"])

        # Find a goal occurring shortly after the shot
        goal_match = goals[
            (
                goals["goal_time"]
                >= shot_time
            )
            &
            (
                goals["goal_time"]
                <= shot_time + GOAL_PROXIMITY_TIME
            )
        ]

        if len(goal_match) > 0:

            outcome = "GOAL"

        else:

            outcome = "NO_GOAL"

        results.append({
            "shot_id": int(shot["shot_id"]),
            "shot_time": shot_time,
            "direction": shot["direction"],
            "peak_speed": float(
                shot["peak_speed"]
            ),
            "peak_confidence": float(
                shot["peak_confidence"]
            ),
            "detections": int(
                shot["detections"]
            ),
            "outcome": outcome
        })

    result = pd.DataFrame(results)

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 40)
    print("SHOT OUTCOME ANALYSIS")
    print("=" * 40)

    print(
        f"Shots analyzed: {len(result)}"
    )

    print()

    print(
        result.to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()