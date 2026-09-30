import os
import subprocess
import pandas as pd


VIDEO_PATH = "data/input/match.mp4"

RESOLVED_FILE = (
    "outputs/events/resolved_event_index.csv"
)

OUTPUT_DIR = (
    "outputs/highlights/shots"
)

BEFORE = 2.0
AFTER = 2.0


def main():

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            VIDEO_PATH
        )

    if not os.path.exists(RESOLVED_FILE):
        raise FileNotFoundError(
            RESOLVED_FILE
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Remove every previous shot clip.
    for filename in os.listdir(OUTPUT_DIR):

        if (
            filename.startswith("shot_")
            and filename.endswith(".mp4")
        ):

            os.remove(
                os.path.join(
                    OUTPUT_DIR,
                    filename
                )
            )

    df = pd.read_csv(
        RESOLVED_FILE
    )

    shots = df[
        df["event_type"] == "SHOT"
    ].copy()

    shots = shots.sort_values(
        "peak_time"
    ).reset_index(
        drop=True
    )

    print("=" * 70)
    print("SHOT CLIP EXTRACTION")
    print("=" * 70)

    print(
        f"Final shots: {len(shots)}"
    )

    created = 0

    for index, row in shots.iterrows():

        peak = float(
            row["peak_time"]
        )

        start = max(
            0.0,
            peak - BEFORE
        )

        end = peak + AFTER

        # Do not let a shot clip contain another
        # final event.
        previous_events = df[
            df["peak_time"] < peak
        ]

        next_events = df[
            df["peak_time"] > peak
        ]

        if not previous_events.empty:

            previous_peak = float(
                previous_events[
                    "peak_time"
                ].max()
            )

            if previous_peak > start:

                start = max(
                    start,
                    previous_peak + 0.10
                )

        if not next_events.empty:

            next_peak = float(
                next_events[
                    "peak_time"
                ].min()
            )

            if next_peak < end:

                end = min(
                    end,
                    next_peak - 0.10
                )

        if end <= start:
            print(
                f"Skipping shot {index + 1}: "
                "no safe clip interval"
            )
            continue

        output = os.path.join(
            OUTPUT_DIR,
            f"shot_{created + 1}.mp4"
        )

        duration = end - start

        print(
            f"Shot {created + 1}: "
            f"{start:.2f}s → {end:.2f}s "
            f"({duration:.2f}s)"
        )

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
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
            output,
        ]

        subprocess.run(
            command,
            check=True
        )

        created += 1

    print()
    print(
        f"Created: {created}"
    )

    print(
        f"Directory: {OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()