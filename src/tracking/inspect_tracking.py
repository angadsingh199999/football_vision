import pandas as pd
from pathlib import Path


# ==================================================
# PATH
# ==================================================

CSV_PATH = Path(
    "outputs/tracking/tracking_data.csv"
)


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(CSV_PATH)

print("================================")
print("TRACKING DATA")
print("================================")

print("Rows:", len(df))
print("Frames:", df["frame"].nunique())

print(
    "Time:",
    round(df["time"].min(), 2),
    "→",
    round(df["time"].max(), 2),
    "seconds"
)


# ==================================================
# CLASS DISTRIBUTION
# ==================================================

print("\n================================")
print("CLASS DISTRIBUTION")
print("================================")

print(
    df["class_name"].value_counts()
)


# ==================================================
# PLAYER TRACKS
# ==================================================

players = df[
    df["class_name"] == "player"
]

print("\n================================")
print("PLAYER TRACKING")
print("================================")

print(
    "Unique player IDs:",
    players["track_id"].nunique()
)

print(
    "Player detections:",
    len(players)
)


# ==================================================
# BALL TRACKING
# ==================================================

ball = df[
    df["class_name"] == "ball"
]

print("\n================================")
print("BALL TRACKING")
print("================================")

print(
    "Ball detections:",
    len(ball)
)

print(
    "Ball frames:",
    ball["frame"].nunique()
)

print(
    "Ball coverage:",
    round(
        ball["frame"].nunique()
        / df["frame"].nunique()
        * 100,
        2
    ),
    "%"
)


# ==================================================
# BALL TRACK IDs
# ==================================================

if len(ball) > 0:

    print(
        "Ball track IDs:",
        sorted(ball["track_id"].unique())
    )


# ==================================================
# BALL CONFIDENCE
# ==================================================

if len(ball) > 0:

    print("\n================================")
    print("BALL CONFIDENCE")
    print("================================")

    print(
        "Mean:",
        round(ball["confidence"].mean(), 3)
    )

    print(
        "Minimum:",
        round(ball["confidence"].min(), 3)
    )

    print(
        "Maximum:",
        round(ball["confidence"].max(), 3)
    )


# ==================================================
# BALL COORDINATES
# ==================================================

if len(ball) > 0:

    print("\n================================")
    print("BALL COORDINATES")
    print("================================")

    print(
        ball[
            [
                "frame",
                "time",
                "track_id",
                "confidence",
                "center_x",
                "center_y"
            ]
        ].head(20).to_string(index=False)
    )


print("\n================================")
print("INSPECTION COMPLETE")
print("================================")