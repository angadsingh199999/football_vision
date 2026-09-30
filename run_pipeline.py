"""Run the football goal-highlights pipeline from the project root.

The Streamlit app writes the uploaded video to ``data/input/match.mp4`` and
uses the same scripts and outputs as this command-line entry point.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_ROOT / "data" / "input" / "match.mp4"

# Keep this list in dependency order. These are the scripts that exist in the
# repository and are also run by app.py.
STEPS = [
    ("Scoreboard OCR", "src/ocr/video_scoreboard_reader.py"),
    ("Score change detection", "src/events/score_change_detector.py"),
    ("Ball and player detection", "src/tracking/ball_detector.py"),
    ("Ball trajectory", "src/tracking/build_ball_trajectory.py"),
    ("Ball movement candidates", "src/events/ball_movement_detector.py"),
    ("Ball movement grouping", "src/events/group_movement_events.py"),
    ("Goal matching", "src/events/goal_event_matcher.py"),
    ("Final goal events", "src/events/finalize_goal_save_events.py"),
    ("Goal highlight clips", "src/events/extract_goal_clips.py"),
]

OUTPUTS_TO_REFRESH = [
    "outputs/scoreboard_readings.json",
    "outputs/score_changes.csv",
    "outputs/goal_candidates.json",
    "outputs/tracking/ball_detections.csv",
    "outputs/tracking/player_detections.csv",
    "outputs/tracking/ball_trajectory.csv",
    "outputs/events/ball_movement_candidates.csv",
    "outputs/events/movement_events.csv",
    "outputs/events/goal_events.csv",
    "outputs/events/final_goal_events.csv",
]


def clear_previous_results() -> None:
    """Remove only generated outputs owned by this goal pipeline."""
    for relative in OUTPUTS_TO_REFRESH:
        path = PROJECT_ROOT / relative
        if path.exists():
            path.unlink()

    clips_dir = PROJECT_ROOT / "outputs" / "highlights" / "goals"
    if clips_dir.exists():
        for clip in clips_dir.glob("goal_*.mp4"):
            clip.unlink()


def main() -> int:
    print("=" * 72)
    print("FOOTBALL VISION — GOAL HIGHLIGHTS AND SCOREBOARD ANALYSIS")
    print(f"Input video: {VIDEO_PATH}")
    print("=" * 72)

    if not VIDEO_PATH.is_file() or VIDEO_PATH.stat().st_size == 0:
        print("ERROR: Put a match video at data/input/match.mp4 first.", file=sys.stderr)
        return 2

    clear_previous_results()

    for number, (name, relative_script) in enumerate(STEPS, start=1):
        script = PROJECT_ROOT / relative_script
        if not script.is_file():
            print(f"ERROR: Required pipeline script is missing: {relative_script}", file=sys.stderr)
            return 2

        print(f"\n[{number}/{len(STEPS)}] {name}", flush=True)
        result = subprocess.run([sys.executable, "-u", str(script)], cwd=PROJECT_ROOT)
        if result.returncode:
            print(f"\nPipeline stopped at: {name} (exit {result.returncode})", file=sys.stderr)
            return result.returncode

    print("\nPipeline completed.")
    print("Goal events: outputs/events/final_goal_events.csv")
    print("Goal clips:  outputs/highlights/goals/")
    print("Scoreboard:  outputs/scoreboard_readings.json")
    print("Score changes: outputs/goal_candidates.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
