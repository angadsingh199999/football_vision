import pandas as pd


SHOTS_CSV = "outputs/events/final_shot_events.csv"
TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"

OUTPUT_CSV = "outputs/events/save_candidates_v2.csv"


# Goal-mouth regions for 1280x720 video
LEFT_GOAL_X = 200
RIGHT_GOAL_X = 1080

GOAL_Y_MIN = 150
GOAL_Y_MAX = 570

LOOK_AHEAD = 5.0

# Ball must get this close to a goal line
GOAL_DISTANCE = 150


def main():

    shots = pd.read_csv(SHOTS_CSV)
    trajectory = pd.read_csv(TRAJECTORY_CSV)

    candidates = []

    for _, shot in shots.iterrows():

        shot_id = int(shot["shot_id"])
        peak_time = float(shot["peak_time"])
        direction = shot["direction"]

        future = trajectory[
            (trajectory["time"] >= peak_time) &
            (trajectory["time"] <= peak_time + LOOK_AHEAD)
        ].copy()

        if future.empty:
            continue

        # Only consider the goal on the direction of the shot
        if direction == "RIGHT":

            future["goal_distance"] = (
                RIGHT_GOAL_X - future["center_x"]
            )

        else:

            future["goal_distance"] = (
                future["center_x"] - LEFT_GOAL_X
            )

        # Keep ball positions moving toward the relevant goal
        near_goal = future[
            (future["goal_distance"] >= 0) &
            (future["goal_distance"] <= GOAL_DISTANCE) &
            (future["center_y"] >= GOAL_Y_MIN) &
            (future["center_y"] <= GOAL_Y_MAX)
        ]

        if near_goal.empty:
            continue

        closest = near_goal.loc[
            near_goal["goal_distance"].idxmin()
        ]

        candidates.append({
            "shot_id": shot_id,
            "shot_time": peak_time,
            "direction": direction,
            "peak_speed": float(shot["peak_speed"]),
            "peak_confidence": float(
                shot["peak_confidence"]
            ),
            "closest_time": float(
                closest["time"]
            ),
            "closest_x": float(
                closest["center_x"]
            ),
            "closest_y": float(
                closest["center_y"]
            ),
            "goal_distance": float(
                closest["goal_distance"]
            )
        })

    result = pd.DataFrame(candidates)

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 40)
    print("SAVE CANDIDATES V2")
    print("=" * 40)

    print(
        f"Candidates: {len(result)}"
    )

    print()

    if not result.empty:
        print(
            result.to_string(index=False)
        )

    print()
    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 40)


if __name__ == "__main__":
    main()