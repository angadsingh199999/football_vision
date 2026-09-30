import pandas as pd
from pathlib import Path


INPUT_CSV = "outputs/events/action_moments.csv"
OUTPUT_CSV = "outputs/events/final_action_candidates.csv"


def main():

    df = pd.read_csv(INPUT_CSV)

    print("=" * 40)
    print("DEDUPLICATE ACTION MOMENTS")
    print("=" * 40)

    print(f"Input moments: {len(df)}")

    if df.empty:
        print("No action moments found.")
        return

    # Calculate a simple strength score
    df["score"] = (
        df["peak_speed"]
        *
        df["peak_confidence"]
    )

    # Strongest moments first
    df = df.sort_values(
        "score",
        ascending=False
    ).reset_index(drop=True)

    selected = []

    for _, candidate in df.iterrows():

        start = float(candidate["start_time"])
        end = float(candidate["end_time"])

        overlaps = False

        for chosen in selected:

            chosen_start = float(
                chosen["start_time"]
            )

            chosen_end = float(
                chosen["end_time"]
            )

            # Check whether the two clips overlap
            if (
                start < chosen_end
                and end > chosen_start
            ):
                overlaps = True
                break

        if not overlaps:
            selected.append(candidate)

    result = pd.DataFrame(selected)

    # Put final candidates back in chronological order
    result = result.sort_values(
        "start_time"
    ).reset_index(drop=True)

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
    print("FINAL ACTION CANDIDATES")
    print("=" * 40)

    print(f"Candidates: {len(result)}")
    print()

    for _, row in result.iterrows():

        print(
            f"Candidate {int(row['candidate_id'])}: "
            f"{row['start_time']:.2f}s → "
            f"{row['end_time']:.2f}s "
            f"| peak={row['peak_time']:.2f}s "
            f"| speed={row['peak_speed']:.2f} "
            f"| score={row['score']:.2f}"
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()