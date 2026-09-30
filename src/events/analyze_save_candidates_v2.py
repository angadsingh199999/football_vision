import pandas as pd


SAVES_CSV = "outputs/events/save_candidates_v2.csv"
TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"

OUTPUT_CSV = "outputs/events/analyzed_save_candidates.csv"

AFTER_WINDOW = 2.0


def main():

    saves = pd.read_csv(SAVES_CSV)
    trajectory = pd.read_csv(TRAJECTORY_CSV)

    results = []

    for _, save in saves.iterrows():

        shot_id = int(save["shot_id"])
        direction = save["direction"]
        closest_time = float(save["closest_time"])

        after = trajectory[
            (trajectory["time"] >= closest_time) &
            (trajectory["time"] <= closest_time + AFTER_WINDOW)
        ].copy()

        if after.empty:
            continue

        start_x = float(save["closest_x"])
        start_y = float(save["closest_y"])

        end = after.iloc[-1]

        end_x = float(end["center_x"])
        end_y = float(end["center_y"])

        dx = end_x - start_x
        dy = end_y - start_y

        distance_after = (dx ** 2 + dy ** 2) ** 0.5

        # Determine whether the ball moved back away
        # from the goal after reaching it.
        if direction == "RIGHT":
            moved_away = end_x < start_x
        else:
            moved_away = end_x > start_x

        results.append({
            "shot_id": shot_id,
            "shot_time": float(save["shot_time"]),
            "direction": direction,
            "closest_time": closest_time,
            "closest_x": start_x,
            "closest_y": start_y,
            "goal_distance": float(save["goal_distance"]),
            "after_end_time": float(end["time"]),
            "after_end_x": end_x,
            "after_end_y": end_y,
            "movement_after": round(distance_after, 2),
            "moved_away_from_goal": moved_away
        })

    result = pd.DataFrame(results)

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 40)
    print("SAVE CANDIDATE ANALYSIS")
    print("=" * 40)

    print(f"Candidates analyzed: {len(result)}")
    print()

    if not result.empty:
        print(result.to_string(index=False))

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()