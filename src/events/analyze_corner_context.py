import pandas as pd
import math


CORNERS_CSV = "outputs/events/final_corner_candidates.csv"
PLAYERS_CSV = "outputs/tracking/tracking_data.csv"

OUTPUT_CSV = "outputs/events/analyzed_corner_candidates.csv"

TIME_WINDOW = 1.0
PLAYER_RADIUS = 150


def distance(x1, y1, x2, y2):
    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


def main():

    corners = pd.read_csv(
        CORNERS_CSV
    )

    players = pd.read_csv(
        PLAYERS_CSV
    )

    results = []

    for _, corner in corners.iterrows():

        corner_id = int(
            corner["corner_id"]
        )

        time = float(
            corner["peak_time"]
        )

        # Step 25 outputs corner_x / corner_y.
        # These are the ball coordinates at the boundary event.
        ball_x = float(
            corner["corner_x"]
        )

        ball_y = float(
            corner["corner_y"]
        )

        nearby = players[
            (players["time"] >= time - TIME_WINDOW) &
            (players["time"] <= time + TIME_WINDOW)
        ]

        distances = []

        for _, player in nearby.iterrows():

            d = distance(
                ball_x,
                ball_y,
                float(player["center_x"]),
                float(player["center_y"])
            )

            distances.append(d)

        if distances:

            distances.sort()

            players_50 = sum(
                d <= 50
                for d in distances
            )

            players_100 = sum(
                d <= 100
                for d in distances
            )

            players_150 = sum(
                d <= PLAYER_RADIUS
                for d in distances
            )

            nearest_player = distances[0]

        else:

            players_50 = 0
            players_100 = 0
            players_150 = 0
            nearest_player = 9999

        results.append({
            "corner_id": corner_id,
            "peak_time": time,
            "boundary": corner["boundary"],
            "corner_x": ball_x,
            "corner_y": ball_y,

            # Step 25 calls this candidate_score,
            # not corner_score.
            "candidate_score": float(
                corner["candidate_score"]
            ),

            "nearest_player": nearest_player,
            "players_within_50": players_50,
            "players_within_100": players_100,
            "players_within_150": players_150
        })

    result = pd.DataFrame(
        results
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 50)
    print("CORNER PLAYER CONTEXT")
    print("=" * 50)

    print(
        f"Candidates analyzed: {len(result)}"
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
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()