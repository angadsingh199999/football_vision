import pandas as pd
from pathlib import Path


INPUT_CSV = "outputs/events/shot_candidates.csv"
OUTPUT_CSV = "outputs/events/final_shot_events.csv"

# Candidates this close in time are considered part of
# the same continuous ball-action sequence.
MAX_EVENT_GAP = 1.80

# After grouping, only merge events whose peaks are very close.
# This is a second protection against duplicate detections.
DEDUP_GAP = 1.25

MIN_SUPPORT_POINTS = 1

MAX_EVENT_DURATION = 3.00


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_groups(df):

    df = df.sort_values("time").reset_index(drop=True)

    groups = []
    current = []

    for _, row in df.iterrows():

        if not current:
            current = [row]
            continue

        previous = current[-1]

        gap = (
            safe_float(row["time"])
            - safe_float(previous["time"])
        )

        # Same continuous movement.
        if gap <= MAX_EVENT_GAP:
            current.append(row)

        else:
            groups.append(current)
            current = [row]

    if current:
        groups.append(current)

    return groups


def summarize_group(rows, event_id):

    group = pd.DataFrame(rows)

    group = group.sort_values(
        "time"
    ).reset_index(drop=True)

    if len(group) < MIN_SUPPORT_POINTS:
        return None

    group["speed"] = pd.to_numeric(
        group["speed"],
        errors="coerce"
    ).fillna(0.0)

    peak_index = group["speed"].idxmax()

    peak = group.loc[peak_index]

    raw_start = safe_float(
        group["start_time"].min()
    )

    raw_end = safe_float(
        group["end_time"].max()
    )

    peak_time = safe_float(
        peak["time"]
    )

    duration = raw_end - raw_start

    if duration > MAX_EVENT_DURATION:

        start_time = max(
            raw_start,
            peak_time - MAX_EVENT_DURATION / 2
        )

        end_time = min(
            raw_end,
            peak_time + MAX_EVENT_DURATION / 2
        )

    else:

        start_time = raw_start
        end_time = raw_end

    return {
        "event_id": event_id,
        "start_time": start_time,
        "end_time": end_time,
        "peak_time": peak_time,
        "peak_speed": safe_float(
            peak["speed"]
        ),
        "peak_confidence": safe_float(
            peak["confidence"]
        ),
        "direction": str(
            peak["direction"]
        ),
        "detections": len(group),
        "distance": safe_float(
            group["distance"].max()
            if "distance" in group.columns
            else 0.0
        ),
        "horizontal_ratio": safe_float(
            group["horizontal_ratio"].max()
            if "horizontal_ratio" in group.columns
            else 0.0
        ),
    }


def event_strength(event):

    return (
        safe_float(event["peak_speed"])
        *
        safe_float(event["peak_confidence"])
    )


def merge_duplicate_events(events):

    if not events:
        return []

    events = sorted(
        events,
        key=lambda x: x["peak_time"]
    )

    final_events = []

    for event in events:

        if not final_events:
            final_events.append(event)
            continue

        previous = final_events[-1]

        peak_gap = abs(
            safe_float(event["peak_time"])
            -
            safe_float(previous["peak_time"])
        )

        same_direction = (
            str(event["direction"])
            ==
            str(previous["direction"])
        )

        # Only merge extremely close peaks that are
        # travelling in the same direction.
        if (
            peak_gap <= DEDUP_GAP
            and same_direction
        ):

            previous_strength = event_strength(
                previous
            )

            current_strength = event_strength(
                event
            )

            if current_strength > previous_strength:

                kept = event
                other = previous

            else:

                kept = previous
                other = event

            kept["start_time"] = min(
                safe_float(kept["start_time"]),
                safe_float(other["start_time"])
            )

            kept["end_time"] = max(
                safe_float(kept["end_time"]),
                safe_float(other["end_time"])
            )

            kept["detections"] = int(
                safe_float(kept["detections"])
                +
                safe_float(other["detections"])
            )

            final_events[-1] = kept

        else:

            final_events.append(event)

    for index, event in enumerate(
        final_events,
        start=1
    ):
        event["event_id"] = index

    return final_events


def main():

    Path(
        OUTPUT_CSV
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(
        INPUT_CSV
    )

    print("=" * 70)
    print("FINAL SHOT EVENTS")
    print("=" * 70)

    if df.empty:

        print(
            "No shot candidates found."
        )

        pd.DataFrame(
            columns=[
                "event_id",
                "start_time",
                "end_time",
                "peak_time",
                "peak_speed",
                "peak_confidence",
                "direction",
                "detections",
                "distance",
                "horizontal_ratio",
            ]
        ).to_csv(
            OUTPUT_CSV,
            index=False
        )

        return

    required = {
        "time",
        "start_time",
        "end_time",
        "speed",
        "confidence",
        "direction",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    groups = build_groups(
        df
    )

    events = []

    for event_id, rows in enumerate(
        groups,
        start=1
    ):

        event = summarize_group(
            rows,
            event_id
        )

        if event is not None:
            events.append(event)

    grouped_count = len(
        events
    )

    events = merge_duplicate_events(
        events
    )

    result = pd.DataFrame(
        events
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"Raw candidates : {len(df)}"
    )

    print(
        f"Grouped events : {grouped_count}"
    )

    print(
        f"Final shots    : {len(result)}"
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
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()