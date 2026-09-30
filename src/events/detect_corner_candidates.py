import pandas as pd
import math


TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"
OUTPUT_CSV = "outputs/events/corner_candidates.csv"

WIDTH = 1280
HEIGHT = 720

# Wider corner region
CORNER_X = 180
CORNER_Y = 150

# Ball must have meaningful movement
MIN_SPEED = 40

# Look for a change in movement direction
MIN_DIRECTION_CHANGE = 30

# Group nearby detections
MAX_GAP = 2.0


def distance(x1, y1, x2, y2):

    return math.sqrt(
        (x1 - x2) ** 2 +
        (y1 - y2) ** 2
    )


def get_corner(x, y):

    if x <= CORNER_X and y <= CORNER_Y:
        return "TOP_LEFT"

    if x >= WIDTH - CORNER_X and y <= CORNER_Y:
        return "TOP_RIGHT"

    if x <= CORNER_X and y >= HEIGHT - CORNER_Y:
        return "BOTTOM_LEFT"

    if x >= WIDTH - CORNER_X and y >= HEIGHT - CORNER_Y:
        return "BOTTOM_RIGHT"

    return None


def main():

    df = pd.read_csv(
        TRAJECTORY_CSV
    )

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    candidates = []

    for i in range(2, len(df)):

        previous = df.iloc[i - 1]
        current = df.iloc[i]

        time = float(
            current["time"]
        )

        x = float(
            current["center_x"]
        )

        y = float(
            current["center_y"]
        )

        prev_x = float(
            previous["center_x"]
        )

        prev_y = float(
            previous["center_y"]
        )

        speed = float(
            current["speed"]
        )

        if speed < MIN_SPEED:
            continue

        corner = get_corner(
            x,
            y
        )

        if corner is None:
            continue

        # Movement vector
        dx = x - prev_x
        dy = y - prev_y

        # Previous movement vector
        prevprev = df.iloc[i - 2]

        dx_prev = (
            prev_x -
            float(prevprev["center_x"])
        )

        dy_prev = (
            prev_y -
            float(prevprev["center_y"])
        )

        # Calculate direction change
        magnitude_1 = math.sqrt(
            dx ** 2 + dy ** 2
        )

        magnitude_2 = math.sqrt(
            dx_prev ** 2 +
            dy_prev ** 2
        )

        if magnitude_1 == 0 or magnitude_2 == 0:
            continue

        dot = (
            dx * dx_prev +
            dy * dy_prev
        )

        cosine = dot / (
            magnitude_1 *
            magnitude_2
        )

        cosine = max(
            -1,
            min(1, cosine)
        )

        angle = math.degrees(
            math.acos(cosine)
        )

        # A substantial direction change near
        # the corner is useful evidence.
        if angle < MIN_DIRECTION_CHANGE:
            continue

        candidates.append({
            "time": time,
            "center_x": x,
            "center_y": y,
            "speed": speed,
            "direction_change": angle,
            "corner": corner
        })

    result = pd.DataFrame(
        candidates
    )

    if result.empty:

        result.to_csv(
            OUTPUT_CSV,
            index=False
        )

        print("=" * 40)
        print("CORNER CANDIDATES V2")
        print("=" * 40)
        print("Candidates: 0")
        print()
        print(
            f"Saved: {OUTPUT_CSV}"
        )
        print("=" * 40)

        return

    # Group nearby candidates
    result = result.sort_values(
        "time"
    ).reset_index(drop=True)

    groups = []
    current_group = []

    for _, row in result.iterrows():

        if not current_group:
            current_group.append(row)
            continue

        previous_time = float(
            current_group[-1]["time"]
        )

        current_time = float(
            row["time"]
        )

        if (
            current_time - previous_time
            <= MAX_GAP
        ):
            current_group.append(row)

        else:
            groups.append(
                current_group
            )
            current_group = [row]

    if current_group:
        groups.append(
            current_group
        )

    final = []

    for corner_id, rows in enumerate(
        groups,
        start=1
    ):

        group = pd.DataFrame(
            rows
        )

        # Pick strongest direction change
        idx = group[
            "direction_change"
        ].idxmax()

        peak = group.loc[idx]

        final.append({
            "corner_id": corner_id,
            "time": float(
                peak["time"]
            ),
            "center_x": float(
                peak["center_x"]
            ),
            "center_y": float(
                peak["center_y"]
            ),
            "speed": float(
                peak["speed"]
            ),
            "direction_change": float(
                peak["direction_change"]
            ),
            "corner": peak["corner"],
            "detections": len(group)
        })

    output = pd.DataFrame(
        final
    )

    output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 40)
    print("CORNER CANDIDATES V2")
    print("=" * 40)

    print(
        f"Candidates: {len(output)}"
    )

    print()

    print(
        output.to_string(
            index=False
        )
    )

    print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 40)


if __name__ == "__main__":
    main()