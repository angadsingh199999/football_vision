import subprocess
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

HIGHLIGHTS_DIR = Path(
    "outputs/highlights"
)

OUTPUT_PATH = (
    HIGHLIGHTS_DIR
    / "final_highlights.mp4"
)


# ============================================================
# FIND GOAL CLIPS
# ============================================================

goal_clips = sorted(
    HIGHLIGHTS_DIR.glob("goal_*.mp4")
)

if not goal_clips:

    print("No goal clips found.")

    raise SystemExit


print("================================")
print("COMBINING HIGHLIGHTS")
print("================================")

print(
    "Goal clips:",
    len(goal_clips)
)

for clip in goal_clips:
    print(" -", clip)


# ============================================================
# CREATE CONCAT FILE
# ============================================================

concat_file = (
    HIGHLIGHTS_DIR
    / "concat_list.txt"
)


with open(
    concat_file,
    "w"
) as f:

    for clip in goal_clips:

        absolute_path = (
            clip.resolve()
        )

        f.write(
            f"file '{absolute_path}'\n"
        )


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if OUTPUT_PATH.exists():

    OUTPUT_PATH.unlink()


# ============================================================
# FFMPEG CONCATENATION
# ============================================================

command = [

    "ffmpeg",

    "-y",

    "-f",
    "concat",

    "-safe",
    "0",

    "-i",
    str(concat_file),

    "-c",
    "copy",

    "-movflags",
    "+faststart",

    str(OUTPUT_PATH)
]


result = subprocess.run(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)


# ============================================================
# CHECK RESULT
# ============================================================

if result.returncode != 0:

    print(
        "FFmpeg combination failed:"
    )

    print(
        result.stderr
    )

    raise RuntimeError(
        "Could not combine goal clips."
    )


# ============================================================
# GET FINAL VIDEO INFORMATION
# ============================================================

probe_command = [

    "ffprobe",

    "-v",
    "error",

    "-show_entries",
    "format=duration",

    "-of",
    "default=noprint_wrappers=1:nokey=1",

    str(OUTPUT_PATH)
]


probe = subprocess.run(
    probe_command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)


duration = float(
    probe.stdout.strip()
)


print()
print("================================")
print("HIGHLIGHTS COMPLETE")
print("================================")

print(
    "Clips:",
    len(goal_clips)
)

print(
    "Duration:",
    round(duration, 2),
    "seconds"
)

print(
    "Saved:",
    OUTPUT_PATH
)

print("================================")