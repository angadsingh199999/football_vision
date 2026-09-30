import os
import subprocess
import pandas as pd


INPUT_VIDEO = "data/input/match.mp4"
INPUT_CSV = "outputs/events/final_corner_events.csv"
OUTPUT_DIR = "outputs/highlights/corners"

CLIP_DURATION = 10
HALF_DURATION = 5

VIDEO_DURATION = 223.68


def extract_clip(start_time, end_time, output_path):

    duration = end_time - start_time

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start_time),
        "-i", INPUT_VIDEO,
        "-t", str(duration),
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path
    ]

    subprocess.run(
        command,
        check=True
    )


def remove_old_corner_clips():

    if not os.path.exists(OUTPUT_DIR):
        return

    for filename in os.listdir(OUTPUT_DIR):

        if (
            filename.startswith("corner_")
            and filename.endswith(".mp4")
        ):

            path = os.path.join(
                OUTPUT_DIR,
                filename
            )

            os.remove(path)

            print(
                f"Removed old clip: {filename}"
            )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Remove stale clips from previous runs.
    # --------------------------------------------------------

    remove_old_corner_clips()

    # --------------------------------------------------------
    # Check input CSV.
    # --------------------------------------------------------

    if not os.path.exists(INPUT_CSV):

        raise FileNotFoundError(
            f"Corner event file not found: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV
    )

    # --------------------------------------------------------
    # No confirmed corners.
    # --------------------------------------------------------

    if df.empty:

        print("=" * 60)
        print("EXTRACTING CORNER CLIPS")
        print("=" * 60)

        print(
            "No confirmed corner events found."
        )

        print(
            "No corner clips created."
        )

        print(
            f"Output directory cleaned: {OUTPUT_DIR}"
        )

        print("=" * 60)

        return

    # --------------------------------------------------------
    # Validate required columns.
    # --------------------------------------------------------

    required_columns = {
        "final_corner_id",
        "peak_time",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "final_corner_events.csv is missing "
            "columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Extract clips.
    # --------------------------------------------------------

    print("=" * 60)
    print("EXTRACTING CORNER CLIPS")
    print("=" * 60)

    created = 0

    for _, row in df.iterrows():

        corner_id = int(
            row["final_corner_id"]
        )

        peak_time = float(
            row["peak_time"]
        )

        start_time = max(
            0.0,
            peak_time - HALF_DURATION
        )

        end_time = min(
            peak_time + HALF_DURATION,
            VIDEO_DURATION
        )

        if end_time <= start_time:

            print(
                f"Skipping corner {corner_id}: "
                "invalid clip range."
            )

            continue

        output_path = os.path.join(
            OUTPUT_DIR,
            f"corner_{corner_id}.mp4"
        )

        print(
            f"Corner {corner_id}: "
            f"{start_time:.2f}s -> "
            f"{end_time:.2f}s"
        )

        extract_clip(
            start_time,
            end_time,
            output_path
        )

        created += 1

    print()
    print(
        f"Created {created} corner clips"
    )

    print(
        f"Saved in: {OUTPUT_DIR}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()