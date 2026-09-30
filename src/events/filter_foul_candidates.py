import pandas as pd


INPUT_CSV = "outputs/events/foul_candidates.csv"
OUTPUT_CSV = "outputs/events/final_foul_candidates.csv"


# Minimum foul score
MIN_FOUL_SCORE = 0.50

# With the revised detector, nearby_players represents
# UNIQUE player track IDs, not raw detections.
MIN_NEARBY_PLAYERS = 2

# Maximum time gap between consecutive candidate detections
MAX_TIME_GAP = 1.0

# Minimum number of ball observations supporting an event
MIN_BALL_OBSERVATIONS = 3.0


def main():

    print("=" * 50)
    print("FILTERING FOUL CANDIDATES")
    print("=" * 50)

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        print("No foul candidates found.")
        return

    required_columns = [
        "foul_candidate_id",
        "time",
        "ball_x",
        "ball_y",
        "nearby_players",
        "nearest_player",
        "speed_change",
        "foul_score",
        "ball_observations"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------

    numeric_columns = [
        "time",
        "ball_x",
        "ball_y",
        "nearby_players",
        "nearest_player",
        "speed_change",
        "foul_score",
        "ball_observations"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=numeric_columns
    ).copy()

    if df.empty:
        print("No valid candidates after numeric validation.")
        return

    # --------------------------------------------------
    # BASIC FILTER
    # --------------------------------------------------

    df = df[
        (df["foul_score"] >= MIN_FOUL_SCORE)
        &
        (df["nearby_players"] >= MIN_NEARBY_PLAYERS)
        &
        (df["ball_observations"] >= MIN_BALL_OBSERVATIONS)
    ].copy()

    if df.empty:
        print("No candidates survived basic filtering.")

        # Important: overwrite stale output
        pd.DataFrame().to_csv(
            OUTPUT_CSV,
            index=False
        )

        return

    # --------------------------------------------------
    # SORT BY TIME
    # --------------------------------------------------

    df = df.sort_values(
        "time"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # GROUP TEMPORALLY CLOSE CANDIDATES
    # --------------------------------------------------

    events = []

    current = []
    previous_time = None

    for _, row in df.iterrows():

        current_time = float(
            row["time"]
        )

        if previous_time is None:

            current = [row]

        elif (
            current_time - previous_time
            <= MAX_TIME_GAP
        ):

            current.append(row)

        else:

            events.append(
                pd.DataFrame(current)
            )

            current = [row]

        previous_time = current_time

    if current:
        events.append(
            pd.DataFrame(current)
        )

    # --------------------------------------------------
    # SELECT STRONGEST REPRESENTATIVE
    # --------------------------------------------------

    final_events = []

    for event in events:

        start_time = float(
            event["time"].min()
        )

        end_time = float(
            event["time"].max()
        )

        # Pick strongest candidate in this temporal group
        best = event.loc[
            event["foul_score"].idxmax()
        ]

        final_events.append({
            "peak_time": float(
                best["time"]
            ),

            "start_time": start_time,

            "end_time": end_time,

            "ball_x": float(
                best["ball_x"]
            ),

            "ball_y": float(
                best["ball_y"]
            ),

            "nearby_players": int(
                best["nearby_players"]
            ),

            "nearest_player": float(
                best["nearest_player"]
            ),

            "speed_change": float(
                best["speed_change"]
            ),

            "foul_score": float(
                best["foul_score"]
            ),

            "ball_observations": int(
                best["ball_observations"]
            ),

            "detections": len(event)
        })

    result = pd.DataFrame(
        final_events
    )

    if result.empty:
        print("No final foul candidates.")

        pd.DataFrame().to_csv(
            OUTPUT_CSV,
            index=False
        )

        return

    # --------------------------------------------------
    # SORT BY SCORE
    # --------------------------------------------------

    result = result.sort_values(
        "foul_score",
        ascending=False
    ).reset_index(drop=True)

    result["foul_candidate_id"] = range(
        1,
        len(result) + 1
    )

    columns = [
        "foul_candidate_id",
        "peak_time",
        "start_time",
        "end_time",
        "ball_x",
        "ball_y",
        "nearby_players",
        "nearest_player",
        "speed_change",
        "foul_score",
        "ball_observations",
        "detections"
    ]

    result = result[columns]

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print()
    print(
        f"Candidates after basic filtering: "
        f"{len(df)}"
    )

    print(
        f"Final foul candidates: "
        f"{len(result)}"
    )

    print()

    print(
        result.to_string(index=False)
    )

    print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()