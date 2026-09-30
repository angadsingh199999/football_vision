import pandas as pd
from pathlib import Path


INPUT_CSV = "outputs/tracking/ball_trajectory.csv"
OUTPUT_CSV = "outputs/events/ball_movement_candidates_v2.csv"

SPEED_THRESHOLD = 80.0
MIN_CONFIDENCE = 0.20


def main():

    df = pd.read_csv(INPUT_CSV)

    print("=" * 40)
    print("BALL MOVEMENT DETECTOR")
    print("=" * 40)

    print(f"Trajectory rows: {len(df)}")

    # Keep reliable ball detections
    df = df[df["confidence"] >= MIN_CONFIDENCE].copy()

    # Find fast ball movements
    candidates = df[
        df["speed"] >= SPEED_THRESHOLD
    ].copy()

    print(
        f"Speed threshold: "
        f"{SPEED_THRESHOLD} pixels/frame"
    )

    print(
        f"Movement candidates: "
        f"{len(candidates)}"
    )

    if len(candidates) > 0:

        print()
        print("Candidate moments:")

        for _, row in candidates.iterrows():

            print(
                f"  {row['time']:.2f}s "
                f"| speed={row['speed']:.2f} "
                f"| confidence={row['confidence']:.2f}"
            )

    Path(OUTPUT_CSV).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    candidates.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()