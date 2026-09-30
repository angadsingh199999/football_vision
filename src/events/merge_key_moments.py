import pandas as pd
from pathlib import Path


INPUT_CSV = "outputs/events/key_moment_candidates.csv"
OUTPUT_CSV = "outputs/events/key_moments.csv"

MERGE_GAP = 2.0


def main():

    df = pd.read_csv(INPUT_CSV)

    print("=" * 40)
    print("MERGE KEY MOMENTS")
    print("=" * 40)

    print(f"Candidates: {len(df)}")

    if df.empty:
        print("No candidates found.")
        return

    # Sort chronologically
    df = df.sort_values("clip_start").reset_index(drop=True)

    events = []

    current = {
        "start": float(df.iloc[0]["clip_start"]),
        "end": float(df.iloc[0]["clip_end"]),
        "peak_time": float(df.iloc[0]["peak_time"]),
        "peak_speed": float(df.iloc[0]["peak_speed"]),
        "peak_confidence": float(df.iloc[0]["peak_confidence"]),
        "candidate_count": 1
    }

    for i in range(1, len(df)):

        row = df.iloc[i]

        start = float(row["clip_start"])
        end = float(row["clip_end"])

        # Does this candidate overlap or sit very close?
        if start <= current["end"] + MERGE_GAP:

            current["end"] = max(
                current["end"],
                end
            )

            current["candidate_count"] += 1

            # Keep strongest movement as representative peak
            if float(row["peak_speed"]) > current["peak_speed"]:

                current["peak_time"] = float(
                    row["peak_time"]
                )

                current["peak_speed"] = float(
                    row["peak_speed"]
                )

                current["peak_confidence"] = float(
                    row["peak_confidence"]
                )

        else:

            events.append(current)

            current = {
                "start": start,
                "end": end,
                "peak_time": float(row["peak_time"]),
                "peak_speed": float(row["peak_speed"]),
                "peak_confidence": float(row["peak_confidence"]),
                "candidate_count": 1
            }

    events.append(current)

    rows = []

    for event_id, event in enumerate(events, start=1):

        rows.append({
            "moment_id": event_id,
            "start_time": event["start"],
            "end_time": event["end"],
            "duration": event["end"] - event["start"],
            "peak_time": event["peak_time"],
            "peak_speed": event["peak_speed"],
            "peak_confidence": event["peak_confidence"],
            "candidate_count": event["candidate_count"]
        })

    result = pd.DataFrame(rows)

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
    print("KEY MOMENTS")
    print("=" * 40)

    print(f"Final moments: {len(result)}")
    print()

    for _, row in result.iterrows():

        print(
            f"Moment {int(row['moment_id'])}: "
            f"{row['start_time']:.2f}s → "
            f"{row['end_time']:.2f}s "
            f"| peak={row['peak_time']:.2f}s "
            f"| speed={row['peak_speed']:.2f} "
            f"| candidates={int(row['candidate_count'])}"
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()