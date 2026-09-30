import pandas as pd
import numpy as np


TRACKING_CSV = "outputs/tracking/tracking_data.csv"
TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"

OUTPUT_CSV = "outputs/events/foul_candidates.csv"


# ============================================================
# PARAMETERS
# ============================================================

TIME_WINDOW = 0.4

PLAYER_DISTANCE = 60.0

MIN_UNIQUE_PLAYERS = 2

MIN_BALL_OBSERVATIONS = 3

# Ball must show a meaningful local speed change.
MIN_SPEED_CHANGE = 30.0

# Candidate events must be separated.
MIN_EVENT_GAP = 1.0


# ============================================================
# HELPERS
# ============================================================

def local_speed_change(
    ball,
    event_time,
):

    before = ball[
        (ball["time"] >= event_time - 0.5)
        &
        (ball["time"] < event_time)
    ]

    after = ball[
        (ball["time"] > event_time)
        &
        (ball["time"] <= event_time + 0.5)
    ]

    if before.empty or after.empty:
        return 0.0

    before_speed = pd.to_numeric(
        before["speed"],
        errors="coerce"
    ).dropna()

    after_speed = pd.to_numeric(
        after["speed"],
        errors="coerce"
    ).dropna()

    if before_speed.empty or after_speed.empty:
        return 0.0

    # Use median speeds to reduce individual noisy detections.
    before_median = float(
        before_speed.median()
    )

    after_median = float(
        after_speed.median()
    )

    return abs(
        after_median
        - before_median
    )


def main():

    print("=" * 60)
    print("FOUL CANDIDATE DETECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    tracking = pd.read_csv(
        TRACKING_CSV
    )

    ball = pd.read_csv(
        TRAJECTORY_CSV
    )

    print(
        f"Tracking rows: {len(tracking)}"
    )

    print(
        f"Ball trajectory rows: {len(ball)}"
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    required_tracking = [
        "time",
        "class_name",
        "center_x",
        "center_y",
        "track_id",
    ]

    required_ball = [
        "time",
        "center_x",
        "center_y",
        "speed",
    ]

    missing_tracking = [
        c
        for c in required_tracking
        if c not in tracking.columns
    ]

    missing_ball = [
        c
        for c in required_ball
        if c not in ball.columns
    ]

    if missing_tracking:

        raise ValueError(
            "Missing tracking columns: "
            + str(missing_tracking)
        )

    if missing_ball:

        raise ValueError(
            "Missing trajectory columns: "
            + str(missing_ball)
        )

    # --------------------------------------------------------
    # NUMERIC CONVERSION
    # --------------------------------------------------------

    for column in [
        "time",
        "center_x",
        "center_y",
        "track_id",
    ]:

        tracking[column] = pd.to_numeric(
            tracking[column],
            errors="coerce"
        )

    for column in [
        "time",
        "center_x",
        "center_y",
        "speed",
    ]:

        ball[column] = pd.to_numeric(
            ball[column],
            errors="coerce"
        )

    tracking = tracking.dropna(
        subset=[
            "time",
            "center_x",
            "center_y",
            "track_id",
        ]
    )

    ball = ball.dropna(
        subset=[
            "time",
            "center_x",
            "center_y",
            "speed",
        ]
    )

    # --------------------------------------------------------
    # PLAYERS ONLY
    # --------------------------------------------------------

    players = tracking[
        tracking["class_name"]
        .astype(str)
        .str.lower()
        .eq("player")
    ].copy()

    print(
        f"Player tracking rows: {len(players)}"
    )

    if players.empty:

        print(
            "No player detections found."
        )

        return

    # --------------------------------------------------------
    # PROCESS BALL MOMENTS
    # --------------------------------------------------------

    candidates = []

    for _, ball_row in ball.iterrows():

        event_time = float(
            ball_row["time"]
        )

        ball_x = float(
            ball_row["center_x"]
        )

        ball_y = float(
            ball_row["center_y"]
        )

        # ----------------------------------------------------
        # BALL OBSERVATIONS
        # ----------------------------------------------------

        ball_window = ball[
            (
                ball["time"]
                >= event_time - TIME_WINDOW
            )
            &
            (
                ball["time"]
                <= event_time + TIME_WINDOW
            )
        ]

        if (
            len(ball_window)
            < MIN_BALL_OBSERVATIONS
        ):
            continue

        # ----------------------------------------------------
        # PLAYER OBSERVATIONS
        # ----------------------------------------------------

        nearby = players[
            (
                players["time"]
                >= event_time - TIME_WINDOW
            )
            &
            (
                players["time"]
                <= event_time + TIME_WINDOW
            )
        ].copy()

        if nearby.empty:
            continue

        # ----------------------------------------------------
        # PLAYER DISTANCES
        # ----------------------------------------------------

        nearby["distance"] = np.sqrt(
            (
                nearby["center_x"]
                - ball_x
            ) ** 2
            +
            (
                nearby["center_y"]
                - ball_y
            ) ** 2
        )

        close = nearby[
            nearby["distance"]
            <= PLAYER_DISTANCE
        ].copy()

        if close.empty:
            continue

        # ----------------------------------------------------
        # UNIQUE PLAYERS
        # ----------------------------------------------------

        unique_players = int(
            close["track_id"]
            .dropna()
            .nunique()
        )

        if (
            unique_players
            < MIN_UNIQUE_PLAYERS
        ):
            continue

        # ----------------------------------------------------
        # NEAREST PLAYER
        # ----------------------------------------------------

        nearest_player = float(
            close["distance"].min()
        )

        # ----------------------------------------------------
        # LOCAL BALL SPEED CHANGE
        # ----------------------------------------------------

        speed_change = local_speed_change(
            ball,
            event_time
        )

        if speed_change < MIN_SPEED_CHANGE:
            continue

        # ----------------------------------------------------
        # SCORES
        # ----------------------------------------------------

        proximity_score = max(
            0.0,
            1.0
            - (
                nearest_player
                / PLAYER_DISTANCE
            )
        )

        player_score = min(
            unique_players / 4.0,
            1.0
        )

        speed_score = min(
            speed_change / 150.0,
            1.0
        )

        foul_score = (
            0.35 * proximity_score
            +
            0.25 * player_score
            +
            0.40 * speed_score
        )

        candidates.append({

            "time": event_time,

            "ball_x": ball_x,

            "ball_y": ball_y,

            "nearby_players": unique_players,

            "nearest_player": nearest_player,

            "speed_change": speed_change,

            "foul_score": foul_score,

            "ball_observations": len(
                ball_window
            ),
        })

    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    result = pd.DataFrame(
        candidates
    )

    if result.empty:

        print()
        print(
            "No foul candidates found."
        )

        result.to_csv(
            OUTPUT_CSV,
            index=False
        )

        print(
            f"Saved: {OUTPUT_CSV}"
        )

        print("=" * 60)

        return

    # --------------------------------------------------------
    # NON-MAXIMUM SUPPRESSION
    # --------------------------------------------------------
    #
    # Many neighboring ball rows describe the same physical
    # event. Keep only the strongest candidate in each
    # temporal neighborhood.
    # --------------------------------------------------------

    result = result.sort_values(
        "foul_score",
        ascending=False
    ).reset_index(
        drop=True
    )

    selected = []

    for _, row in result.iterrows():

        event_time = float(
            row["time"]
        )

        too_close = False

        for selected_row in selected:

            selected_time = float(
                selected_row["time"]
            )

            if abs(
                event_time
                - selected_time
            ) < MIN_EVENT_GAP:

                too_close = True
                break

        if not too_close:

            selected.append(
                row.to_dict()
            )

    result = pd.DataFrame(
        selected
    )

    # --------------------------------------------------------
    # SORT CHRONOLOGICALLY
    # --------------------------------------------------------

    result = result.sort_values(
        "time"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # CREATE IDS
    # --------------------------------------------------------

    result.insert(
        0,
        "foul_candidate_id",
        range(
            1,
            len(result) + 1
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print(
        f"Raw qualifying candidates: "
        f"{len(candidates)}"
    )

    print(
        f"After temporal suppression: "
        f"{len(result)}"
    )

    print()

    print(
        result.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()