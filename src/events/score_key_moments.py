import pandas as pd
from pathlib import Path


CANDIDATES_CSV = "outputs/events/final_action_candidates.csv"
PLAYER_CONTEXT_CSV = "outputs/events/player_action_context.csv"

OUTPUT_CSV = "outputs/events/scored_key_moments.csv"


def normalize(value, minimum, maximum):

    if value <= minimum:
        return 0.0

    if value >= maximum:
        return 1.0

    return (value - minimum) / (maximum - minimum)


def main():

    candidates = pd.read_csv(
        CANDIDATES_CSV
    )

    context = pd.read_csv(
        PLAYER_CONTEXT_CSV
    )

    df = candidates.merge(
    context[
        [
            "candidate_id",
            "nearest_player_distance",
            "players_within_100px"
        ]
    ],
    on="candidate_id",
    how="left"
    ) 

    print("=" * 40)
    print("SCORE KEY MOMENTS")
    print("=" * 40)

    results = []

    for _, row in df.iterrows():

        # ------------------------------------------
        # BALL SPEED SCORE
        # ------------------------------------------

        speed_score = normalize(
            float(row["peak_speed"]),
            100,
            250
        )

        # ------------------------------------------
        # BALL CONFIDENCE SCORE
        # ------------------------------------------

        confidence_score = normalize(
            float(row["peak_confidence"]),
            0.20,
            0.80
        )

        # ------------------------------------------
        # PLAYER PROXIMITY SCORE
        # ------------------------------------------

        distance = float(
            row["nearest_player_distance"]
        )

        proximity_score = 1.0 - min(
            distance / 200.0,
            1.0
        )

        # ------------------------------------------
        # PLAYER DENSITY SCORE
        # ------------------------------------------

        density_score = min(
            float(row["players_within_100px"]) / 50.0,
            1.0
        )

        # ------------------------------------------
        # FINAL SCORE
        # ------------------------------------------

        score = (
            speed_score * 0.40
            +
            confidence_score * 0.20
            +
            proximity_score * 0.25
            +
            density_score * 0.15
        )

        results.append({

            "candidate_id":
                int(row["candidate_id"]),

            "start_time":
                float(row["start_time"]),

            "end_time":
                float(row["end_time"]),

            "peak_time":
                float(row["peak_time"]),

            "peak_speed":
                float(row["peak_speed"]),

            "peak_confidence":
                float(row["peak_confidence"]),

            "nearest_player_distance":
                distance,

            "players_within_100px":
                int(row["players_within_100px"]),

            "speed_score":
                speed_score,

            "confidence_score":
                confidence_score,

            "proximity_score":
                proximity_score,

            "density_score":
                density_score,

            "key_moment_score":
                score
        })

    result = pd.DataFrame(results)

    result = result.sort_values(
        "key_moment_score",
        ascending=False
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
    print("RANKED KEY MOMENTS")
    print("=" * 40)

    for rank, (_, row) in enumerate(
        result.iterrows(),
        start=1
    ):

        print(
            f"Rank {rank}: "
            f"Candidate {int(row['candidate_id'])} "
            f"| peak={row['peak_time']:.2f}s "
            f"| score={row['key_moment_score']:.3f}"
        )

        print(
            f"  Speed: {row['peak_speed']:.2f}"
            f" | Confidence: "
            f"{row['peak_confidence']:.2f}"
        )

        print(
            f"  Nearest player: "
            f"{row['nearest_player_distance']:.1f}px"
            f" | Players <=100px: "
            f"{int(row['players_within_100px'])}"
        )

        print()

    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()