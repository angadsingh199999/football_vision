import pandas as pd
import numpy as np
from pathlib import Path


TRACKING_CSV = "outputs/tracking/tracking_data.csv"
CANDIDATES_CSV = "outputs/events/final_action_candidates.csv"
OUTPUT_CSV = "outputs/events/player_action_context.csv"

WINDOW = 0.5


def main():

    tracking = pd.read_csv(TRACKING_CSV)
    candidates = pd.read_csv(CANDIDATES_CSV)

    print("=" * 40)
    print("PLAYER CONTEXT ANALYSIS")
    print("=" * 40)

    # Only players
    players = tracking[
        tracking["class_name"] == "player"
    ].copy()

    print(f"Player detections: {len(players)}")
    print(f"Candidates: {len(candidates)}")

    results = []

    for _, candidate in candidates.iterrows():

        candidate_id = int(
            candidate["candidate_id"]
        )

        peak_time = float(
            candidate["peak_time"]
        )

        # Find closest ball trajectory point
        ball_data = pd.read_csv(
            "outputs/tracking/ball_trajectory.csv"
        )

        ball_data["time_diff"] = (
            ball_data["time"] - peak_time
        ).abs()

        ball = ball_data.loc[
            ball_data["time_diff"].idxmin()
        ]

        ball_x = float(ball["center_x"])
        ball_y = float(ball["center_y"])

        # Find player detections near peak
        nearby_players = players[
            (players["time"] >= peak_time - WINDOW)
            &
            (players["time"] <= peak_time + WINDOW)
        ].copy()

        distances = []

        for _, player in nearby_players.iterrows():

            player_x = float(
                player["center_x"]
            )

            player_y = float(
                player["center_y"]
            )

            distance = np.sqrt(
                (player_x - ball_x) ** 2
                +
                (player_y - ball_y) ** 2
            )

            distances.append(distance)

        if distances:

            distances = np.array(distances)

            nearest_distance = float(
                distances.min()
            )

            players_50 = int(
                np.sum(distances <= 50)
            )

            players_100 = int(
                np.sum(distances <= 100)
            )

            players_150 = int(
                np.sum(distances <= 150)
            )

        else:

            nearest_distance = np.nan
            players_50 = 0
            players_100 = 0
            players_150 = 0

        results.append({

            "candidate_id": candidate_id,

            "peak_time": peak_time,

            "ball_x": ball_x,
            "ball_y": ball_y,

            "nearest_player_distance":
                nearest_distance,

            "players_within_50px":
                players_50,

            "players_within_100px":
                players_100,

            "players_within_150px":
                players_150,

            "peak_speed":
                float(candidate["peak_speed"])
        })

    result = pd.DataFrame(results)

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
    print("PLAYER CONTEXT")
    print("=" * 40)

    for _, row in result.iterrows():

        nearest = row[
            "nearest_player_distance"
        ]

        if pd.isna(nearest):
            nearest_text = "N/A"
        else:
            nearest_text = f"{nearest:.1f}px"

        print(
            f"Candidate "
            f"{int(row['candidate_id'])}"
            f" | peak={row['peak_time']:.2f}s"
        )

        print(
            f"  Ball: "
            f"({row['ball_x']:.1f}, "
            f"{row['ball_y']:.1f})"
        )

        print(
            f"  Nearest player: "
            f"{nearest_text}"
        )

        print(
            f"  Players <=50px: "
            f"{int(row['players_within_50px'])}"
        )

        print(
            f"  Players <=100px: "
            f"{int(row['players_within_100px'])}"
        )

        print(
            f"  Players <=150px: "
            f"{int(row['players_within_150px'])}"
        )

        print()

    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()