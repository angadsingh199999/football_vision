import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_CSV = Path(
    "outputs/tracking/ball_trajectory.csv"
)

OUTPUT_CSV = Path(
    "outputs/events/ball_movement_candidates.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

# Pixels/frame.
# We start conservatively and inspect the result.
SPEED_THRESHOLD = 30.0

# Minimum confidence for a movement candidate.
CONFIDENCE_THRESHOLD = 0.30


# ============================================================
# LOAD TRAJECTORY
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("================================")
print("BALL MOVEMENT DETECTOR")
print("================================")

print("Trajectory rows:", len(df))


# ============================================================
# FILTER
# ============================================================

candidates = df[
    (df["speed"] >= SPEED_THRESHOLD)
    &
    (df["confidence"] >= CONFIDENCE_THRESHOLD)
].copy()


# ============================================================
# ADD TIME
# ============================================================

# The tracking CSV contains time in seconds.
# Keep it as the event timestamp.

candidates = candidates.sort_values(
    "frame"
).reset_index(drop=True)


# ============================================================
# OUTPUT COLUMNS
# ============================================================

columns = [
    "frame",
    "time",
    "center_x",
    "center_y",
    "speed",
    "direction",
    "confidence",
    "segment_id"
]

candidates = candidates[
    [c for c in columns if c in candidates.columns]
]


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

candidates.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("Speed threshold:", SPEED_THRESHOLD)
print(
    "Movement candidates:",
    len(candidates)
)

print()
print(candidates.to_string(index=False))

print()
print("Saved:", OUTPUT_CSV)

print("================================")