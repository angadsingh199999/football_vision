import pandas as pd


INPUT_CSV = "outputs/events/filtered_shot_events.csv"
OUTPUT_CSV = "outputs/events/final_shot_events.csv"

MERGE_TIME_GAP = 3.0


def main():

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        print("No shot events found.")
        return

    df = df.sort_values("peak_time").reset_index(drop=True)

    final_events = []

    current = None

    for _, row in df.iterrows():

        if current is None:
            current = row.to_dict()
            continue

        time_gap = (
            float(row["peak_time"])
            - float(current["peak_time"])
        )

        if time_gap <= MERGE_TIME_GAP:

            # Keep the stronger detection
            if (
                float(row["peak_speed"])
                * float(row["peak_confidence"])
                >
                float(current["peak_speed"])
                * float(current["peak_confidence"])
            ):
                current = row.to_dict()

        else:
            final_events.append(current)
            current = row.to_dict()

    if current is not None:
        final_events.append(current)

    result = pd.DataFrame(final_events)

    result["shot_id"] = range(
        1,
        len(result) + 1
    )

    columns = [
        "shot_id"
    ] + [
        c for c in result.columns
        if c != "shot_id"
    ]

    result = result[columns]

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 40)
    print("FINAL SHOT EVENTS")
    print("=" * 40)

    print(
        f"Input candidates: {len(df)}"
    )

    print(
        f"Final shot events: {len(result)}"
    )

    print()

    print(
        result.to_string(index=False)
    )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()