import pandas as pd


INPUT_CSV = "outputs/events/shot_events.csv"
OUTPUT_CSV = "outputs/events/filtered_shot_events.csv"

MIN_DETECTIONS = 2
MIN_SPEED = 100
MIN_CONFIDENCE = 0.25


def main():

    df = pd.read_csv(INPUT_CSV)

    filtered = df[
        (df["detections"] >= MIN_DETECTIONS)
        & (df["peak_speed"] >= MIN_SPEED)
        & (df["peak_confidence"] >= MIN_CONFIDENCE)
    ].copy()

    filtered = filtered.sort_values(
        "peak_time"
    ).reset_index(drop=True)

    filtered["shot_id"] = range(
        1,
        len(filtered) + 1
    )

    # Put shot_id first
    columns = [
        "shot_id"
    ] + [
        c for c in filtered.columns
        if c != "shot_id"
    ]

    filtered = filtered[columns]

    filtered.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 40)
    print("FILTERED SHOT EVENTS")
    print("=" * 40)

    print(
        f"Input events: {len(df)}"
    )

    print(
        f"Accepted shots: {len(filtered)}"
    )

    print()

    if len(filtered) > 0:
        print(
            filtered.to_string(index=False)
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()