import pandas as pd
import numpy as np
from pathlib import Path


TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"
CANDIDATES_CSV = "outputs/events/final_action_candidates.csv"
OUTPUT_CSV = "outputs/events/action_analysis.csv"


def main():

    trajectory = pd.read_csv(TRAJECTORY_CSV)
    candidates = pd.read_csv(CANDIDATES_CSV)

    print("=" * 40)
    print("ANALYZE ACTION CANDIDATES")
    print("=" * 40)

    print(f"Trajectory rows: {len(trajectory)}")
    print(f"Candidates: {len(candidates)}")

    results = []

    for _, candidate in candidates.iterrows():

        start_time = float(candidate["start_time"])
        end_time = float(candidate["end_time"])

        peak_time = float(candidate["peak_time"])

        # Get trajectory inside candidate window
        window = trajectory[
            (trajectory["time"] >= start_time)
            &
            (trajectory["time"] <= end_time)
        ].copy()

        if len(window) < 2:
            continue

        # Sort chronologically
        window = window.sort_values("time")

        first = window.iloc[0]
        last = window.iloc[-1]

        start_x = float(first["center_x"])
        start_y = float(first["center_y"])

        end_x = float(last["center_x"])
        end_y = float(last["center_y"])

        dx = end_x - start_x
        dy = end_y - start_y

        distance = np.sqrt(
            dx ** 2 +
            dy ** 2
        )

        # Horizontal direction
        if dx > 50:
            direction = "right"
        elif dx < -50:
            direction = "left"
        else:
            direction = "vertical"

        results.append({
            "candidate_id": int(
                candidate["candidate_id"]
            ),

            "start_time": start_time,
            "end_time": end_time,
            "peak_time": peak_time,

            "start_x": start_x,
            "start_y": start_y,

            "end_x": end_x,
            "end_y": end_y,

            "delta_x": dx,
            "delta_y": dy,

            "distance": distance,

            "direction": direction,

            "peak_speed": float(
                candidate["peak_speed"]
            ),

            "peak_confidence": float(
                candidate["peak_confidence"]
            )
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
    print("ACTION ANALYSIS")
    print("=" * 40)

    for _, row in result.iterrows():

        print(
            f"Candidate {int(row['candidate_id'])}: "
            f"{row['start_time']:.2f}s → "
            f"{row['end_time']:.2f}s"
        )

        print(
            f"  Start: "
            f"({row['start_x']:.1f}, "
            f"{row['start_y']:.1f})"
        )

        print(
            f"  End:   "
            f"({row['end_x']:.1f}, "
            f"{row['end_y']:.1f})"
        )

        print(
            f"  ΔX: {row['delta_x']:.1f} "
            f"| ΔY: {row['delta_y']:.1f}"
        )

        print(
            f"  Distance: {row['distance']:.1f} "
            f"| Direction: {row['direction']}"
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