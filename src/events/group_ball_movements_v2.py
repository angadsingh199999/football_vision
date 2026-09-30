import pandas as pd
from pathlib import Path


INPUT_CSV = "outputs/events/ball_movement_candidates_v2.csv"
OUTPUT_CSV = "outputs/events/ball_movement_events_v2.csv"

MAX_GAP = 1.5


def main():

    df = pd.read_csv(INPUT_CSV)

    print("=" * 40)
    print("GROUP BALL MOVEMENTS")
    print("=" * 40)

    print(f"Candidates: {len(df)}")

    if df.empty:
        print("No movement candidates found.")
        return

    df = df.sort_values("time").reset_index(drop=True)

    events = []

    current = [df.iloc[0]]

    for i in range(1, len(df)):

        previous_time = float(df.iloc[i - 1]["time"])
        current_time = float(df.iloc[i]["time"])

        gap = current_time - previous_time

        if gap <= MAX_GAP:
            current.append(df.iloc[i])

        else:
            events.append(current)
            current = [df.iloc[i]]

    events.append(current)

    rows = []

    for event_id, event in enumerate(events, start=1):

        event_df = pd.DataFrame(event)

        peak_row = event_df.loc[
            event_df["speed"].idxmax()
        ]

        rows.append({
            "event_id": event_id,
            "start_time": event_df["time"].min(),
            "end_time": event_df["time"].max(),
            "duration": (
                event_df["time"].max()
                -
                event_df["time"].min()
            ),
            "peak_speed": peak_row["speed"],
            "peak_time": peak_row["time"],
            "peak_confidence": peak_row["confidence"],
            "detections": len(event_df)
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
    print("MOVEMENT EVENTS")
    print("=" * 40)

    print(f"Events: {len(result)}")
    print()

    for _, row in result.iterrows():

        print(
            f"Event {int(row['event_id'])}: "
            f"{row['start_time']:.2f}s → "
            f"{row['end_time']:.2f}s "
            f"| peak={row['peak_speed']:.2f} "
            f"| detections={int(row['detections'])}"
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()