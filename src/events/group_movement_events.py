import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

INPUT_CSV = Path(
    "outputs/events/ball_movement_candidates.csv"
)

OUTPUT_CSV = Path(
    "outputs/events/movement_events.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

# Two movement detections within this many seconds
# are considered part of the same movement episode.
MERGE_GAP_SECONDS = 2.0


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("================================")
print("MOVEMENT EVENT GROUPING")
print("================================")

print("Candidates:", len(df))

if df.empty:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=[
        "event_id", "start_time", "end_time", "peak_speed",
        "peak_time", "peak_x", "peak_y", "max_confidence",
        "detections", "duration",
    ]).to_csv(OUTPUT_CSV, index=False)
    print("No movement candidates; wrote an empty movement-events table.")
    raise SystemExit(0)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    "time"
).reset_index(drop=True)


# ============================================================
# GROUP EVENTS
# ============================================================

events = []

current = None


for _, row in df.iterrows():

    time = float(row["time"])

    if current is None:

        current = {
            "start_time": time,
            "end_time": time,
            "peak_speed": float(row["speed"]),
            "peak_time": time,
            "peak_x": float(row["center_x"]),
            "peak_y": float(row["center_y"]),
            "max_confidence": float(row["confidence"]),
            "detections": 1
        }

        continue


    gap = time - current["end_time"]


    # --------------------------------------------------------
    # Same event
    # --------------------------------------------------------

    if gap <= MERGE_GAP_SECONDS:

        current["end_time"] = time

        current["detections"] += 1

        if float(row["speed"]) > current["peak_speed"]:

            current["peak_speed"] = float(row["speed"])
            current["peak_time"] = time
            current["peak_x"] = float(row["center_x"])
            current["peak_y"] = float(row["center_y"])


        current["max_confidence"] = max(
            current["max_confidence"],
            float(row["confidence"])
        )


    # --------------------------------------------------------
    # New event
    # --------------------------------------------------------

    else:

        events.append(current)

        current = {
            "start_time": time,
            "end_time": time,
            "peak_speed": float(row["speed"]),
            "peak_time": time,
            "peak_x": float(row["center_x"]),
            "peak_y": float(row["center_y"]),
            "max_confidence": float(row["confidence"]),
            "detections": 1
        }


# Add final event

if current is not None:
    events.append(current)


# ============================================================
# DATAFRAME
# ============================================================

events_df = pd.DataFrame(events)

events_df.insert(
    0,
    "event_id",
    range(1, len(events_df) + 1)
)


# ============================================================
# EVENT DURATION
# ============================================================

events_df["duration"] = (
    events_df["end_time"]
    - events_df["start_time"]
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

events_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# PRINT
# ============================================================

print()
print(
    "Grouped movement events:",
    len(events_df)
)

print()
print(events_df.to_string(index=False))

print()
print("Saved:", OUTPUT_CSV)

print("================================")
