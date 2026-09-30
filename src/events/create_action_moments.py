import pandas as pd
from pathlib import Path


INPUT_CSV = "outputs/events/ball_movement_candidates_v2.csv"
OUTPUT_CSV = "outputs/events/action_moments.csv"

PEAK_GAP = 2.0
BEFORE = 5.0
AFTER = 5.0
MIN_SPEED = 100.0


def main():

    df = pd.read_csv(INPUT_CSV)

    print("=" * 40)
    print("CREATE ACTION MOMENTS")
    print("=" * 40)

    print(f"Movement candidates: {len(df)}")

    # Remove weak movements
    df = df[
        df["speed"] >= MIN_SPEED
    ].copy()

    # Sort by time
    df = df.sort_values("time").reset_index(drop=True)

    print(f"After speed filtering: {len(df)}")

    if df.empty:
        print("No candidates found.")
        return

    moments = []

    current = [df.iloc[0]]

    for i in range(1, len(df)):

        previous_peak = float(
            df.iloc[i - 1]["time"]
        )

        current_peak = float(
            df.iloc[i]["time"]
        )

        gap = current_peak - previous_peak

        if gap <= PEAK_GAP:
            current.append(df.iloc[i])

        else:
            moments.append(current)
            current = [df.iloc[i]]

    moments.append(current)

    rows = []

    for moment_id, group in enumerate(moments, start=1):

        moment_df = pd.DataFrame(group)

        peak_row = moment_df.loc[
            moment_df["speed"].idxmax()
        ]

        peak_time = float(
            peak_row["time"]
        )

        start_time = max(
            0,
            peak_time - BEFORE
        )

        end_time = peak_time + AFTER

        rows.append({
            "moment_id": moment_id,
            "start_time": start_time,
            "end_time": end_time,
            "peak_time": peak_time,
            "peak_speed": float(
                peak_row["speed"]
            ),
            "peak_confidence": float(
                peak_row["confidence"]
            ),
            "detections": len(moment_df)
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
    print("ACTION MOMENTS")
    print("=" * 40)

    print(f"Moments: {len(result)}")
    print()

    for _, row in result.iterrows():

        print(
            f"Moment {int(row['moment_id'])}: "
            f"{row['start_time']:.2f}s → "
            f"{row['end_time']:.2f}s "
            f"| peak={row['peak_time']:.2f}s "
            f"| speed={row['peak_speed']:.2f} "
            f"| detections={int(row['detections'])}"
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()