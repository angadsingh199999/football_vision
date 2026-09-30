import os
import pandas as pd


INPUT_FILE = (
    "outputs/events/resolved_event_index.csv"
)

OUTPUT_FILE = (
    "outputs/highlights/final_highlight_index.csv"
)


def clip_path(event_type, number):

    if event_type == "GOAL":
        return (
            f"outputs/highlights/"
            f"key_moments/goal_{number}.mp4"
        )

    if event_type == "SAVE":
        return (
            f"outputs/highlights/"
            f"saves/save_{number}.mp4"
        )

    if event_type == "SHOT":
        return (
            f"outputs/highlights/"
            f"shots/shot_{number}.mp4"
        )

    return ""


def main():

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            INPUT_FILE
        )

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    allowed = {
        "GOAL",
        "SAVE",
        "SHOT",
    }

    df = df[
        df["event_type"].isin(allowed)
    ].copy()

    df = df.sort_values(
        "peak_time"
    ).reset_index(
        drop=True
    )

    counters = {
        "GOAL": 0,
        "SAVE": 0,
        "SHOT": 0,
    }

    rows = []

    for moment_id, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        event_type = str(
            row["event_type"]
        )

        counters[event_type] += 1

        number = counters[event_type]

        priority = {
            "GOAL": 100,
            "SAVE": 95,
            "SHOT": 80,
        }[event_type]

        rows.append(
            {
                "moment_id": moment_id,
                "type": event_type,
                "event_id": int(
                    row["event_id"]
                ),
                "time": float(
                    row["peak_time"]
                ),
                "start_time": float(
                    row["start_time"]
                ),
                "end_time": float(
                    row["end_time"]
                ),
                "priority": priority,
                "score": float(
                    row["confidence"]
                ),
                "supporting_events": row.get(
                    "supporting_events",
                    ""
                ),
                "resolution_reason": row.get(
                    "resolution_reason",
                    ""
                ),
                "clip": clip_path(
                    event_type,
                    number
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("=" * 70)
    print("FINAL HIGHLIGHT INDEX")
    print("=" * 70)

    print(
        f"Final highlights: {len(result)}"
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
        f"Saved: {OUTPUT_FILE}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()