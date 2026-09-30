import pandas as pd
from pathlib import Path


SCORED_CSV = "outputs/events/scored_key_moments.csv"
GOALS_CSV = "outputs/events/goal_events.csv"

OUTPUT_CSV = "outputs/events/classified_key_moments.csv"


GOAL_TIME_TOLERANCE = 1.0


def main():

    moments = pd.read_csv(SCORED_CSV)
    goals = pd.read_csv(GOALS_CSV)

    print("=" * 40)
    print("CLASSIFY KEY MOMENTS")
    print("=" * 40)

    print(f"Moments: {len(moments)}")
    print(f"Goals: {len(goals)}")

    results = []

    for _, row in moments.iterrows():

        peak_time = float(
            row["peak_time"]
        )

        speed = float(
            row["peak_speed"]
        )

        confidence = float(
            row["peak_confidence"]
        )

        nearest_player = float(
            row["nearest_player_distance"]
        )

        players_near = int(
            row["players_within_100px"]
        )

        # ------------------------------------------
        # CHECK WHETHER THIS IS A GOAL
        # ------------------------------------------

        is_goal = False

        for _, goal in goals.iterrows():

            goal_time = float(
                goal["goal_time"]
            )

            if abs(peak_time - goal_time) <= GOAL_TIME_TOLERANCE:
                is_goal = True
                break

        # ------------------------------------------
        # CLASSIFICATION
        # ------------------------------------------

        if is_goal:

            event_type = "GOAL"

        elif (
            speed >= 180
            and nearest_player <= 50
        ):

            event_type = "SHOT"

        elif (
            speed >= 150
            and nearest_player <= 100
        ):

            event_type = "ATTACK"

        else:

            event_type = "OTHER"

        # ------------------------------------------
        # EVENT SCORE
        # ------------------------------------------

        if event_type == "GOAL":
            priority = 100

        elif event_type == "SHOT":
            priority = 80

        elif event_type == "ATTACK":
            priority = 60

        else:
            priority = 20

        results.append({

            "candidate_id":
                int(row["candidate_id"]),

            "peak_time":
                peak_time,

            "start_time":
                float(row["start_time"]),

            "end_time":
                float(row["end_time"]),

            "event_type":
                event_type,

            "priority":
                priority,

            "key_moment_score":
                float(row["key_moment_score"]),

            "peak_speed":
                speed,

            "peak_confidence":
                confidence,

            "nearest_player_distance":
                nearest_player,

            "players_within_100px":
                players_near
        })

    result = pd.DataFrame(results)

    # Highest priority first,
    # then strongest key-moment score
    result = result.sort_values(
        ["priority", "key_moment_score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    Path(OUTPUT_CSV).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print("=" * 40)
    print("CLASSIFIED MOMENTS")
    print("=" * 40)

    for _, row in result.iterrows():

        print(
            f"Candidate "
            f"{int(row['candidate_id'])}"
            f" | {row['event_type']}"
            f" | peak={row['peak_time']:.2f}s"
            f" | priority={int(row['priority'])}"
            f" | score={row['key_moment_score']:.3f}"
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()