import pandas as pd
import subprocess
from pathlib import Path


VIDEO_PATH = "data/input/match.mp4"

CLASSIFIED_CSV = (
    "outputs/events/classified_key_moments.csv"
)

GOALS_CSV = (
    "outputs/events/goal_events.csv"
)

OUTPUT_DIR = Path(
    "outputs/highlights/key_moments"
)


def extract_clip(
    start_time,
    end_time,
    output_path
):

    duration = end_time - start_time

    command = [
        "ffmpeg",
        "-y",

        "-ss",
        str(start_time),

        "-i",
        VIDEO_PATH,

        "-t",
        str(duration),

        "-map",
        "0:v:0",

        "-map",
        "0:a:0?",

        "-c:v",
        "libx264",

        "-preset",
        "fast",

        "-crf",
        "18",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-movflags",
        "+faststart",

        str(output_path)
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    classified = pd.read_csv(
        CLASSIFIED_CSV
    )

    goals = pd.read_csv(
        GOALS_CSV
    )

    print("=" * 40)
    print("KEY MOMENT CLIPS")
    print("=" * 40)

    clip_number = 1

    # --------------------------------------------------
    # GOALS
    # --------------------------------------------------

    for _, goal in goals.iterrows():

        goal_time = float(
            goal["goal_time"]
        )

        start = max(
            0,
            goal_time - 15
        )

        end = goal_time + 10

        output = OUTPUT_DIR / (
            f"goal_{clip_number}.mp4"
        )

        print(
            f"Goal {clip_number}: "
            f"{start:.2f}s → {end:.2f}s"
        )

        extract_clip(
            start,
            end,
            output
        )

        clip_number += 1

    # --------------------------------------------------
    # NON-GOAL EVENTS
    # --------------------------------------------------

    non_goals = classified[
        classified["event_type"].isin(
            ["SHOT", "ATTACK"]
        )
    ].copy()

    # Chronological order
    non_goals = non_goals.sort_values(
        "peak_time"
    )

    counters = {
        "SHOT": 1,
        "ATTACK": 1
    }

    for _, event in non_goals.iterrows():

        event_type = event["event_type"]

        peak_time = float(
            event["peak_time"]
        )

        start = max(
            0,
            peak_time - 5
        )

        end = peak_time + 5

        name = event_type.lower()

        number = counters[event_type]

        output = OUTPUT_DIR / (
            f"{name}_{number}.mp4"
        )

        print(
            f"{event_type} {number}: "
            f"{start:.2f}s → {end:.2f}s"
        )

        extract_clip(
            start,
            end,
            output
        )

        counters[event_type] += 1

    print()
    print("=" * 40)
    print("CLIP EXTRACTION COMPLETE")
    print("=" * 40)

    files = sorted(
        OUTPUT_DIR.glob("*.mp4")
    )

    print(
        f"Clips created: {len(files)}"
    )

    for file in files:
        print(f"  {file}")

    print("=" * 40)


if __name__ == "__main__":
    main()