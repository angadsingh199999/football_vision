import pandas as pd
import math


TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"
PLAYERS_CSV = "outputs/tracking/tracking_data.csv"

OUTPUT_CSV = "outputs/events/dangerous_attack_candidates.csv"


# Video dimensions
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Goal positions
LEFT_GOAL_X = 0
RIGHT_GOAL_X = FRAME_WIDTH

# Attacking-third boundary
ATTACKING_THIRD = FRAME_WIDTH * 0.67

# Minimum ball speed
MIN_SPEED = 60

# Minimum movement toward goal
MIN_FORWARD_MOVEMENT = 80

# Player context
PLAYER_RADIUS = 150

# Time grouping
MAX_GAP = 2.0


def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


def count_nearby_players(
    players,
    time,
    ball_x,
    ball_y
):

    # Small time window around ball position
    nearby_time = players[
        (players["time"] >= time - 0.5) &
        (players["time"] <= time + 0.5)
    ]

    if nearby_time.empty:
        return 0

    count = 0

    for _, player in nearby_time.iterrows():

        d = distance(
            ball_x,
            ball_y,
            float(player["center_x"]),
            float(player["center_y"])
        )

        if d <= PLAYER_RADIUS:
            count += 1

    return count


def main():

    trajectory = pd.read_csv(
        TRAJECTORY_CSV
    )

    players = pd.read_csv(
        PLAYERS_CSV
    )

    trajectory = trajectory.sort_values(
        "time"
    ).reset_index(drop=True)

    candidates = []

    for i in range(1, len(trajectory)):

        current = trajectory.iloc[i]
        previous = trajectory.iloc[i - 1]

        time = float(current["time"])

        x = float(current["center_x"])
        y = float(current["center_y"])

        prev_x = float(previous["center_x"])

        speed = float(current["speed"])

        if speed < MIN_SPEED:
            continue

        # Determine direction of attack
        if x >= FRAME_WIDTH / 2:

            # Ball is on right half.
            # Moving right = toward right goal.
            forward_movement = x - prev_x
            attacking_goal = "RIGHT"

        else:

            # Ball is on left half.
            # Moving left = toward left goal.
            forward_movement = prev_x - x
            attacking_goal = "LEFT"

        if forward_movement < MIN_FORWARD_MOVEMENT:
            continue

        # Check attacking third
        dangerous_zone = False

        if attacking_goal == "RIGHT":
            if x >= ATTACKING_THIRD:
                dangerous_zone = True

        else:
            if x <= FRAME_WIDTH - ATTACKING_THIRD:
                dangerous_zone = True

        if not dangerous_zone:
            continue

        nearby_players = count_nearby_players(
            players,
            time,
            x,
            y
        )

        # Require player involvement
        if nearby_players < 2:
            continue

        candidates.append({
            "time": time,
            "center_x": x,
            "center_y": y,
            "speed": speed,
            "forward_movement": forward_movement,
            "attacking_goal": attacking_goal,
            "nearby_players": nearby_players
        })

    result = pd.DataFrame(
        candidates
    )

    if result.empty:

        print("=" * 40)
        print("DANGEROUS ATTACK DETECTION")
        print("=" * 40)
        print("No candidates found.")
        print(
            f"Saved: {OUTPUT_CSV}"
        )
        return

    # Group nearby detections into attacks

    result = result.sort_values(
        "time"
    ).reset_index(drop=True)

    attacks = []

    current = []

    for _, row in result.iterrows():

        if not current:

            current.append(row)
            continue

        previous_time = float(
            current[-1]["time"]
        )

        current_time = float(
            row["time"]
        )

        if (
            current_time - previous_time
            <= MAX_GAP
        ):

            current.append(row)

        else:

            attacks.append(
                current
            )

            current = [row]

    if current:
        attacks.append(
            current
        )

    final_attacks = []

    for attack_id, rows in enumerate(
        attacks,
        start=1
    ):

        attack_df = pd.DataFrame(
            rows
        )

        peak_idx = attack_df[
            "speed"
        ].idxmax()

        peak = attack_df.loc[
            peak_idx
        ]

        start_time = float(
            attack_df["time"].min()
        )

        end_time = float(
            attack_df["time"].max()
        )

        peak_speed = float(
            peak["speed"]
        )

        max_players = int(
            attack_df[
                "nearby_players"
            ].max()
        )

        attacking_goal = (
            peak["attacking_goal"]
        )

        final_attacks.append({
            "attack_id": attack_id,
            "start_time": start_time,
            "end_time": end_time,
            "peak_time": float(
                peak["time"]
            ),
            "peak_speed": peak_speed,
            "attacking_goal": attacking_goal,
            "max_nearby_players": max_players,
            "detections": len(
                attack_df
            )
        })

    final = pd.DataFrame(
        final_attacks
    )

    # Rank attacks
    final["attack_score"] = (
        final["peak_speed"] / 250
    ) * 0.6 + (
        final["max_nearby_players"] / 10
    ).clip(upper=1) * 0.4

    final = final.sort_values(
        "attack_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    final.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 40)
    print("DANGEROUS ATTACK DETECTION")
    print("=" * 40)

    print(
        f"Attack candidates: {len(final)}"
    )

    print()

    print(
        final.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 40)


if __name__ == "__main__":
    main()