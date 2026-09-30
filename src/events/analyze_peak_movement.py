import pandas as pd
import numpy as np
from pathlib import Path


TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"
CANDIDATES_CSV = "outputs/events/final_action_candidates.csv"
OUTPUT_CSV = "outputs/events/peak_movement_analysis.csv"

WINDOW = 2.0


def nearest_row(df, target_time):

    if df.empty:
        return None

    index = (
        (df["time"] - target_time)
        .abs()
        .idxmin()
    )

    return df.loc[index]


def main():

    trajectory = pd.read_csv(TRAJECTORY_CSV)
    candidates = pd.read_csv(CANDIDATES_CSV)

    print("=" * 40)
    print("PEAK MOVEMENT ANALYSIS")
    print("=" * 40)

    results = []

    for _, candidate in candidates.iterrows():

        peak_time = float(candidate["peak_time"])

        before = trajectory[
            (trajectory["time"] >= peak_time - WINDOW)
            &
            (trajectory["time"] < peak_time)
        ]

        after = trajectory[
            (trajectory["time"] > peak_time)
            &
            (trajectory["time"] <= peak_time + WINDOW)
        ]

        peak = nearest_row(
            trajectory,
            peak_time
        )

        if peak is None:
            continue

        before_row = nearest_row(
            before,
            peak_time - WINDOW
        )

        after_row = nearest_row(
            after,
            peak_time + WINDOW
        )

        if before_row is None or after_row is None:
            continue

        before_x = float(before_row["center_x"])
        before_y = float(before_row["center_y"])

        peak_x = float(peak["center_x"])
        peak_y = float(peak["center_y"])

        after_x = float(after_row["center_x"])
        after_y = float(after_row["center_y"])

        before_to_peak = np.sqrt(
            (peak_x - before_x) ** 2
            +
            (peak_y - before_y) ** 2
        )

        peak_to_after = np.sqrt(
            (after_x - peak_x) ** 2
            +
            (after_y - peak_y) ** 2
        )

        results.append({

            "candidate_id": int(
                candidate["candidate_id"]
            ),

            "peak_time": peak_time,

            "before_x": before_x,
            "before_y": before_y,

            "peak_x": peak_x,
            "peak_y": peak_y,

            "after_x": after_x,
            "after_y": after_y,

            "before_to_peak_distance":
                before_to_peak,

            "peak_to_after_distance":
                peak_to_after,

            "peak_speed":
                float(candidate["peak_speed"]),

            "peak_confidence":
                float(candidate["peak_confidence"])
        })

    result = pd.DataFrame(results)

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
    print("PEAK ANALYSIS")
    print("=" * 40)

    for _, row in result.iterrows():

        print(
            f"Candidate {int(row['candidate_id'])} "
            f"| peak={row['peak_time']:.2f}s"
        )

        print(
            f"  Before: "
            f"({row['before_x']:.1f}, "
            f"{row['before_y']:.1f})"
        )

        print(
            f"  Peak:   "
            f"({row['peak_x']:.1f}, "
            f"{row['peak_y']:.1f})"
        )

        print(
            f"  After:  "
            f"({row['after_x']:.1f}, "
            f"{row['after_y']:.1f})"
        )

        print(
            f"  Before → Peak: "
            f"{row['before_to_peak_distance']:.1f}px"
        )

        print(
            f"  Peak → After: "
            f"{row['peak_to_after_distance']:.1f}px"
        )

        print(
            f"  Peak speed: "
            f"{row['peak_speed']:.2f}"
        )

        print()

    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()