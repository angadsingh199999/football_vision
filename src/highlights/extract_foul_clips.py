import os
import subprocess
import pandas as pd


INPUT_VIDEO = "data/input/match.mp4"
INPUT_CSV = "outputs/events/filtered_foul_candidates.csv"
OUTPUT_DIR = "outputs/highlights/fouls"

CLIP_HALF_DURATION = 5
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


def main():

    print("=" * 50)
    print("EXTRACTING FOUL CLIPS")
    print("=" * 50)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    df = pd.read_csv(
        INPUT_CSV
    )

    if df.empty:
        print("No foul candidates found.")
        return

    required_columns = [
        "final_foul_id",
        "peak_time"
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

    for _, row in df.iterrows():

        foul_id = int(
            row["final_foul_id"]
        )

        peak_time = float(
            row["peak_time"]
        )

        start_time = max(
            0.0,
            peak_time - CLIP_HALF_DURATION
        )

        end_time = min(
            VIDEO_DURATION,
            peak_time + CLIP_HALF_DURATION
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            f"foul_{foul_id}.mp4"
        )

        print(
            f"Foul candidate {foul_id}: "
            f"{start_time:.2f}s -> "
            f"{end_time:.2f}s"
        )

        extract_clip(
            start_time,
            end_time,
            output_path
        )

    print()
    print(
        f"Created {len(df)} foul clip(s)"
    )

    print(
        f"Saved in: {OUTPUT_DIR}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()