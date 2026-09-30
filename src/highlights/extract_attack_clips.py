import os
import subprocess
import pandas as pd


VIDEO_PATH = "data/input/match.mp4"
ATTACKS_CSV = "outputs/events/final_dangerous_attacks.csv"

OUTPUT_DIR = "outputs/highlights/attacks"

BEFORE = 5.0
AFTER = 5.0

VIDEO_DURATION = 223.68


def extract_clip(start, end, output_path):

    duration = end - start

    command = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-i", VIDEO_PATH,
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
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )


def remove_old_attack_clips():

    if not os.path.exists(OUTPUT_DIR):
        return

    for filename in os.listdir(OUTPUT_DIR):

        if (
            filename.startswith("attack_")
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

    remove_old_attack_clips()

    # --------------------------------------------------------
    # Validate input.
    # --------------------------------------------------------

    if not os.path.exists(ATTACKS_CSV):

        raise FileNotFoundError(
            f"Attack event file not found: {ATTACKS_CSV}"
        )

    attacks = pd.read_csv(
        ATTACKS_CSV
    )

    print("=" * 60)
    print("EXTRACTING ATTACK CLIPS")
    print("=" * 60)

    if attacks.empty:

        print(
            "No final dangerous attacks found."
        )

        print(
            "No attack clips created."
        )

        print(
            f"Output directory cleaned: {OUTPUT_DIR}"
        )

        print("=" * 60)

        return

    # --------------------------------------------------------
    # Validate schema.
    # --------------------------------------------------------

    required_columns = {
        "final_attack_id",
        "peak_time",
    }

    missing = (
        required_columns
        - set(attacks.columns)
    )

    if missing:

        raise ValueError(
            "final_dangerous_attacks.csv is missing "
            "columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # --------------------------------------------------------
    # Extract.
    # --------------------------------------------------------

    created = 0

    for _, attack in attacks.iterrows():

        attack_id = int(
            attack["final_attack_id"]
        )

        peak_time = float(
            attack["peak_time"]
        )

        start = max(
            0.0,
            peak_time - BEFORE
        )

        end = min(
            VIDEO_DURATION,
            peak_time + AFTER
        )

        if end <= start:

            print(
                f"Skipping attack {attack_id}: "
                "invalid clip range."
            )

            continue

        output_path = os.path.join(
            OUTPUT_DIR,
            f"attack_{attack_id}.mp4"
        )

        print(
            f"Attack {attack_id}: "
            f"{start:.2f}s → {end:.2f}s"
        )

        extract_clip(
            start,
            end,
            output_path
        )

        created += 1

    print()
    print(
        f"Created: {created}"
    )

    print(
        f"Directory: {OUTPUT_DIR}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()