import pandas as pd
import numpy as np


TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"
OUTPUT_CSV = "outputs/events/save_candidates.csv"


# Video dimensions
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720


# Approximate goal zones
LEFT_GOAL_X = 100
RIGHT_GOAL_X = 1180

GOAL_Y_MIN = 200
GOAL_Y_MAX = 520


MIN_SPEED = 100
MAX_TIME_GAP = 1.0


def main():

    df = pd.read_csv(TRAJECTORY_CSV)

    df = df.sort_values("time").reset_index(drop=True)

    candidates = []

    for i in range(1, len(df) - 1):

        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]

        # Require continuous trajectory
        if curr["time"] - prev["time"] > MAX_TIME_GAP:
            continue

        if nxt["time"] - curr["time"] > MAX_TIME_GAP:
            continue

        speed = float(curr["speed"])

        if speed < MIN_SPEED:
            continue

        x = float(curr["center_x"])
        y = float(curr["center_y"])

        # Check whether ball is near a goal
        near_left_goal = (
            x <= LEFT_GOAL_X
            and GOAL_Y_MIN <= y <= GOAL_Y_MAX
        )

        near_right_goal = (
            x >= RIGHT_GOAL_X
            and GOAL_Y_MIN <= y <= GOAL_Y_MAX
        )

        if not (near_left_goal or near_right_goal):
            continue

        # Ball movement before and after current point
        dx_before = (
            float(curr["center_x"])
            - float(prev["center_x"])
        )

        dx_after = (
            float(nxt["center_x"])
            - float(curr["center_x"])
        )

        # Direction reversal
        direction_reversal = (
            dx_before * dx_after < 0
        )

        if not direction_reversal:
            continue

        candidates.append({
            "time": float(curr["time"]),
            "x": x,
            "y": y,
            "speed": speed,
            "confidence": float(curr["confidence"]),
            "goal_side": (
                "LEFT"
                if near_left_goal
                else "RIGHT"
            ),
            "dx_before": dx_before,
            "dx_after": dx_after
        })

    result = pd.DataFrame(candidates)

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 40)
    print("SAVE CANDIDATES")
    print("=" * 40)

    print(
        f"Candidates: {len(result)}"
    )

    if len(result) > 0:
        print()
        print(
            result.to_string(index=False)
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()