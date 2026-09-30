import os
import pandas as pd


REQUIRED_FILES = [

    "data/input/match.mp4",

    "models/best.pt",

    "outputs/tracking/tracking_data.csv",

    "outputs/tracking/ball_detections.csv",

    "outputs/tracking/ball_trajectory.csv",

    "outputs/scoreboard_readings.json",

    "outputs/events/goal_events.csv",

    "outputs/events/final_shot_events.csv",

    "outputs/events/final_save_events.csv",

    "outputs/events/resolved_event_index.csv",

    "outputs/highlights/final_highlight_index.csv",

    "outputs/highlights/highlight_clip_validation.csv",
]


ALLOWED_TYPES = {
    "GOAL",
    "SAVE",
    "SHOT",
}


def main():

    print()
    print("=" * 70)
    print("FOOTBALL VISION PIPELINE VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # FILE CHECK
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING REQUIRED FILES")
    print("=" * 70)

    missing = []

    for path in REQUIRED_FILES:

        if os.path.exists(path):

            print(
                f"[OK]      {path}"
            )

        else:

            print(
                f"[MISSING] {path}"
            )

            missing.append(path)

    if missing:

        print()
        print(
            f"Missing files: {len(missing)}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # RESOLVED EVENT INDEX
    # --------------------------------------------------------

    resolved_file = (
        "outputs/events/"
        "resolved_event_index.csv"
    )

    final_file = (
        "outputs/highlights/"
        "final_highlight_index.csv"
    )

    resolved = pd.read_csv(
        resolved_file
    )

    final = pd.read_csv(
        final_file
    )

    print()
    print("=" * 70)
    print("CHECKING RESOLVED EVENTS")
    print("=" * 70)

    print(
        f"Resolved events: "
        f"{len(resolved)}"
    )

    if "event_type" not in resolved.columns:

        print(
            "[FAIL] resolved index "
            "missing event_type"
        )

        raise SystemExit(1)

    invalid_types = set(
        resolved["event_type"].dropna()
    ) - ALLOWED_TYPES

    if invalid_types:

        print(
            "[FAIL] Invalid event types:"
        )

        print(
            sorted(invalid_types)
        )

        raise SystemExit(1)

    print(
        "[OK] Only GOAL/SAVE/SHOT "
        "are present"
    )

    # --------------------------------------------------------
    # CHECK TIMESTAMPS
    # --------------------------------------------------------

    print()
    print(
        "Checking event timestamps..."
    )

    for _, row in resolved.iterrows():

        start = float(
            row["start_time"]
        )

        peak = float(
            row["peak_time"]
        )

        end = float(
            row["end_time"]
        )

        if start < 0:

            raise SystemExit(
                "[FAIL] Negative start time"
            )

        if not (
            start
            <= peak
            <= end
        ):

            raise SystemExit(
                "[FAIL] Invalid event "
                "timestamp ordering"
            )

    print(
        "[OK] Event timestamps valid"
    )

    # --------------------------------------------------------
    # CHECK FOR OVERLAPPING FINAL EVENTS
    # --------------------------------------------------------

    print()
    print(
        "Checking event separation..."
    )

    sorted_events = resolved.sort_values(
        "peak_time"
    ).reset_index(
        drop=True
    )

    for i in range(
        len(sorted_events) - 1
    ):

        current = sorted_events.iloc[i]
        next_event = sorted_events.iloc[i + 1]

        current_end = float(
            current["end_time"]
        )

        next_start = float(
            next_event["start_time"]
        )

        if current_end > next_start:

            print(
                "[FAIL] Overlapping events:"
            )

            print(
                current.to_dict()
            )

            print(
                next_event.to_dict()
            )

            raise SystemExit(1)

    print(
        "[OK] Final event intervals "
        "do not overlap"
    )

    # --------------------------------------------------------
    # FINAL HIGHLIGHT INDEX
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CHECKING FINAL HIGHLIGHT INDEX")
    print("=" * 70)

    print(
        f"Final highlights: "
        f"{len(final)}"
    )

    required_columns = {
        "moment_id",
        "type",
        "event_id",
        "time",
        "start_time",
        "end_time",
        "clip",
    }

    missing_columns = (
        required_columns
        - set(final.columns)
    )

    if missing_columns:

        print(
            "[FAIL] Missing final index "
            "columns:"
        )

        print(
            sorted(missing_columns)
        )

        raise SystemExit(1)

    invalid_final_types = set(
        final["type"].dropna()
    ) - ALLOWED_TYPES

    if invalid_final_types:

        print(
            "[FAIL] Invalid final "
            "highlight types:"
        )

        print(
            sorted(invalid_final_types)
        )

        raise SystemExit(1)

    print(
        "[OK] Final index contains "
        "only GOAL/SAVE/SHOT"
    )

    # --------------------------------------------------------
    # CHECK CLIPS
    # --------------------------------------------------------

    print()
    print(
        "Checking individual clips..."
    )

    missing_clips = []

    for _, row in final.iterrows():

        clip = str(
            row["clip"]
        )

        if not os.path.exists(clip):

            missing_clips.append(
                clip
            )

    if missing_clips:

        print(
            "[FAIL] Missing clips:"
        )

        for clip in missing_clips:
            print(
                f"  {clip}"
            )

        raise SystemExit(1)

    print(
        f"[OK] All {len(final)} "
        "highlight clips exist"
    )

    # --------------------------------------------------------
    # NO COMBINED VIDEO
    # --------------------------------------------------------

    combined_candidates = [
        "outputs/highlights/"
        "combined_highlights.mp4",

        "outputs/highlights/"
        "all_highlights.mp4",
    ]

    combined_found = [
        path
        for path in combined_candidates
        if os.path.exists(path)
    ]

    if combined_found:

        print(
            "[FAIL] Combined highlight "
            "video found:"
        )

        for path in combined_found:
            print(path)

        raise SystemExit(1)

    print(
        "[OK] No combined highlight "
        "video required"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION RESULT")
    print("=" * 70)

    counts = (
        final["type"]
        .value_counts()
        .to_dict()
    )

    print(
        f"Goals : {counts.get('GOAL', 0)}"
    )

    print(
        f"Saves : {counts.get('SAVE', 0)}"
    )

    print(
        f"Shots : {counts.get('SHOT', 0)}"
    )

    print()

    print(
        "PIPELINE VALIDATION PASSED"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()