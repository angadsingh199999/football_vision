import pandas as pd
from pathlib import Path


INPUT_CSV = "outputs/events/ball_movement_events_v2.csv"
OUTPUT_CSV = "outputs/events/key_moment_candidates.csv"

BEFORE = 5.0
AFTER = 5.0

MIN_PEAK_SPEED = 100.0


def main():

    df = pd.read_csv(INPUT_CSV)

    print("=" * 40)
    print("KEY MOMENT CANDIDATES")
    print("=" * 40)

    print(f"Movement events: {len(df)}")

    # Remove very weak movement events
    df = df[
        df["peak_speed"] >= MIN_PEAK_SPEED
    ].copy()

    candidates = []

    for _, row in df.iterrows():

        peak_time = float(row["peak_time"])

        start_time = max(
            0,
            peak_time - BEFORE
        )

        end_time = peak_time + AFTER

        # Simple ranking score
        score = (
            float(row["peak_speed"])
            *
            float(row["peak_confidence"])
        )

        candidates.append({
            "candidate_id": len(candidates) + 1,
            "event_id": int(row["event_id"]),
            "peak_time": peak_time,
            "clip_start": start_time,
            "clip_end": end_time,
            "peak_speed": float(row["peak_speed"]),
            "peak_confidence": float(row["peak_confidence"]),
            "movement_detections": int(row["detections"]),
            "movement_score": score
        })

    result = pd.DataFrame(candidates)

    # Highest scoring moments first
    result = result.sort_values(
        "movement_score",
        ascending=False
    ).reset_index(drop=True)

    # Re-number candidates after sorting
    result["candidate_id"] = range(
        1,
        len(result) + 1
    )

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
    print("CANDIDATES")
    print("=" * 40)

    print(f"Candidates: {len(result)}")
    print()

    for _, row in result.iterrows():

        print(
            f"Candidate {int(row['candidate_id'])}: "
            f"{row['clip_start']:.2f}s → "
            f"{row['clip_end']:.2f}s "
            f"| peak={row['peak_time']:.2f}s "
            f"| speed={row['peak_speed']:.2f} "
            f"| score={row['movement_score']:.2f}"
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()