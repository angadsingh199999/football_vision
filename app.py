import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

VIDEO_PATH = INPUT_DIR / "match.mp4"
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
SCOREBOARD_CONFIG = PROJECT_ROOT / "config" / "scoreboard.json"
ANALYSIS_MANIFEST = OUTPUT_DIR / "analysis_manifest.json"

# Main outputs
BALL_DETECTIONS = (
    OUTPUT_DIR
    / "tracking"
    / "ball_detections.csv"
)

PLAYER_DETECTIONS = (
    OUTPUT_DIR
    / "tracking"
    / "player_detections.csv"
)

BALL_TRAJECTORY = (
    OUTPUT_DIR
    / "tracking"
    / "ball_trajectory.csv"
)

MOVEMENT_CANDIDATES = (
    OUTPUT_DIR
    / "events"
    / "ball_movement_candidates.csv"
)

MOVEMENT_EVENTS = (
    OUTPUT_DIR
    / "events"
    / "movement_events.csv"
)

SCOREBOARD_OUTPUT = (
    OUTPUT_DIR
    / "scoreboard_readings.json"
)

SCORE_CHANGES_OUTPUT = (
    OUTPUT_DIR
    / "score_changes.csv"
)

GOAL_CANDIDATES = (
    OUTPUT_DIR
    / "goal_candidates.json"
)

GOAL_EVENTS = (
    OUTPUT_DIR
    / "events"
    / "goal_events.csv"
)

FINAL_GOAL_EVENTS = (
    OUTPUT_DIR
    / "events"
    / "final_goal_events.csv"
)

GOAL_CLIPS_DIR = (
    OUTPUT_DIR
    / "highlights"
    / "goals"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Football Vision",
    page_icon="⚽",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .goal-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">⚽ Football Vision</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered football goal highlights and scoreboard analysis"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Pipeline")

st.sidebar.write(
    """
### Detection
- YOLO ball detection
- Player detection
- Ball trajectory

### Ball Movement
- Movement candidate detection
- Movement event grouping

### Scoreboard
- RapidOCR
- Match clock
- Teams
- Scores
- Score-change detection

### Goal Detection
- Scoreboard confirmation
- Ball movement localization
- Goal event matching
- Final goal events

### Output
- Goal highlight clips
- Score changes
- OCR timeline
"""
)

try:
    scoreboard_config = json.loads(SCOREBOARD_CONFIG.read_text())
except (OSError, json.JSONDecodeError):
    scoreboard_config = {
        "x1": 45, "y1": 45, "x2": 340, "y2": 90,
        "reference_width": 1280, "reference_height": 720,
    }

with st.sidebar.expander("Scoreboard crop settings"):
    st.caption(
        "Include both team labels, the score and the match clock in the "
        "rectangle. The OCR stage scales it to the uploaded video's resolution "
        "and falls back to a broad top-left crop if needed."
    )
    reference_width = int(scoreboard_config.get("reference_width", 1280))
    reference_height = int(scoreboard_config.get("reference_height", 720))
    crop_col1, crop_col2 = st.columns(2)
    crop_x1 = crop_col1.number_input("Left (x1)", min_value=0, max_value=reference_width, value=int(scoreboard_config["x1"]), step=5)
    crop_y1 = crop_col2.number_input("Top (y1)", min_value=0, max_value=reference_height, value=int(scoreboard_config["y1"]), step=5)
    crop_x2 = crop_col1.number_input("Right (x2)", min_value=1, max_value=reference_width, value=int(scoreboard_config["x2"]), step=5)
    crop_y2 = crop_col2.number_input("Bottom (y2)", min_value=1, max_value=reference_height, value=int(scoreboard_config["y2"]), step=5)


# ============================================================
# HELPER: RUN SCRIPT WITH LIVE OUTPUT
# ============================================================

def run_script(script_path: Path, log_placeholder=None):
    """
    Run a project script using the same Python interpreter
    as Streamlit.

    Uses Python -u so stdout is not buffered.
    Displays live output so Streamlit does not appear frozen.
    """

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found:\n{script_path}"
        )

    command = [
        sys.executable,
        "-u",
        str(script_path),
    ]

    env = os.environ.copy()

    python_paths = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "src"),
        str(PROJECT_ROOT / "src" / "ocr"),
    ]

    existing_pythonpath = env.get("PYTHONPATH")

    if existing_pythonpath:
        python_paths.append(existing_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(
        python_paths
    )

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []

    if process.stdout is not None:

        for line in process.stdout:

            line = line.rstrip()

            if not line:
                continue

            output_lines.append(line)

            # Keep Streamlit display manageable.
            # Only show the most recent 60 lines.
            if log_placeholder is not None:

                recent_lines = output_lines[-60:]

                log_placeholder.code(
                    "\n".join(recent_lines)
                )

    return_code = process.wait()

    return {
        "returncode": return_code,
        "stdout": "\n".join(output_lines),
        "stderr": "",
        "args": command,
    }


# ============================================================
# HELPER: DISPLAY SCRIPT ERROR
# ============================================================

def show_script_error(
    title,
    result,
):

    st.error(title)

    with st.expander(
        "Show technical details",
        expanded=True,
    ):

        st.markdown("### Command")

        st.code(
            " ".join(result["args"])
        )

        st.markdown("### Output")

        if result["stdout"].strip():

            st.code(
                result["stdout"]
            )

        else:

            st.write(
                "No output was produced."
            )

        st.markdown("### Return code")

        st.code(
            str(result["returncode"])
        )


# ============================================================
# HELPER: CLEAN PREVIOUS PIPELINE OUTPUTS
# ============================================================

def clean_previous_outputs():

    files_to_remove = [

        # Tracking
        BALL_DETECTIONS,
        PLAYER_DETECTIONS,
        BALL_TRAJECTORY,

        # Movement
        MOVEMENT_CANDIDATES,
        MOVEMENT_EVENTS,

        # Scoreboard
        SCOREBOARD_OUTPUT,
        GOAL_CANDIDATES,
        SCORE_CHANGES_OUTPUT,

        # Goal events
        GOAL_EVENTS,
        FINAL_GOAL_EVENTS,
    ]

    for file_path in files_to_remove:

        try:

            if file_path.exists():
                file_path.unlink()

        except Exception:
            pass


    # Remove old goal clips
    if GOAL_CLIPS_DIR.exists():

        for file_path in GOAL_CLIPS_DIR.glob(
            "*.mp4"
        ):

            try:

                file_path.unlink()

            except Exception:
                pass

    else:

        GOAL_CLIPS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# HELPER: VIDEO VALIDATION
# ============================================================

def validate_video(video_path):

    if not video_path.exists():

        return False, (
            "Uploaded video does not exist:\n"
            f"{video_path}"
        )

    if video_path.stat().st_size == 0:

        return False, (
            "Uploaded video exists but "
            "its size is 0 bytes."
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        return False, (
            "OpenCV could not open "
            "the uploaded video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    if fps <= 0:

        return False, (
            "Invalid video FPS."
        )

    if frame_count <= 0:

        return False, (
            "Video contains no readable frames."
        )

    return True, {
        "fps": fps,
        "frames": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps,
    }


# ============================================================
# HELPER: EXECUTE PIPELINE STEP
# ============================================================

def execute_step(
    progress,
    status,
    log_placeholder,
    step_number,
    total_steps,
    message,
    script_path,
):

    status.info(
        f"Step {step_number}/{total_steps} — {message}"
    )

    log_placeholder.empty()

    result = run_script(
        script_path,
        log_placeholder,
    )

    if result["returncode"] != 0:

        show_script_error(
            f"{message} failed.",
            result,
        )

        st.stop()

    progress.progress(
        step_number / total_steps
    )

    status.success(
        f"Step {step_number}/{total_steps} completed — "
        f"{message}"
    )

    return result


# ============================================================
# HELPER: CHECK REQUIRED OUTPUT
# ============================================================

def require_output(
    path,
    description,
):

    if not path.exists():

        st.error(
            f"{description} was not created."
        )

        st.code(
            str(path)
        )

        st.stop()


# ============================================================
# LOAD GOAL EVENTS
# ============================================================

def load_goal_events():

    if not FINAL_GOAL_EVENTS.exists():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            FINAL_GOAL_EVENTS
        )

        if df.empty:

            return pd.DataFrame()

        return df

    except Exception:

        return pd.DataFrame()


# ============================================================
# LOAD OCR
# ============================================================

def load_ocr():

    if not SCOREBOARD_OUTPUT.exists():

        return []

    try:

        with open(
            SCOREBOARD_OUTPUT,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(data, list):

            return data

        return []

    except Exception:

        return []


# ============================================================
# FORMAT SCOREBOARD
# ============================================================

def format_scoreboard(row):

    clock = row.get(
        "clock"
    )

    home = row.get(
        "home_team"
    )

    away = row.get(
        "away_team"
    )

    home_score = row.get(
        "home_score"
    )

    away_score = row.get(
        "away_score"
    )

    if clock is None:
        clock = "?"

    if home is None:
        home = "?"

    if away is None:
        away = "?"

    if home_score is None:
        home_score = "?"

    if away_score is None:
        away_score = "?"

    return (
        f"{clock} | "
        f"{home} {home_score} - "
        f"{away_score} {away}"
    )


# ============================================================
# UPLOAD
# ============================================================

st.header(
    "1. Upload Match Video"
)

uploaded_video = st.file_uploader(
    "Upload a football match video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv",
        "mpeg4",
    ],
)

# Bind every displayed result to the exact uploaded bytes, not just the fixed
# internal path data/input/match.mp4. This prevents a previous upload's scores
# and clips from being presented as results for a newly selected video.
uploaded_video_sha256 = None
if uploaded_video is not None:
    uploaded_video_sha256 = hashlib.sha256(
        uploaded_video.getbuffer()
    ).hexdigest()

result_files_for_current_upload = (
    FINAL_GOAL_EVENTS,
    SCOREBOARD_OUTPUT,
    SCORE_CHANGES_OUTPUT,
)
manifest_video_sha256 = None
if ANALYSIS_MANIFEST.is_file():
    try:
        manifest_video_sha256 = json.loads(
            ANALYSIS_MANIFEST.read_text(encoding="utf-8")
        ).get("video_sha256")
    except (OSError, json.JSONDecodeError):
        pass

has_cached_results_for_upload = bool(
    uploaded_video_sha256
    and all(path.is_file() for path in result_files_for_current_upload)
    and (
        st.session_state.get("analyzed_video_sha256") == uploaded_video_sha256
        or manifest_video_sha256 == uploaded_video_sha256
    )
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded_video:

    st.success(
        f"Uploaded: {uploaded_video.name}"
    )

    st.caption(
        "A new upload starts analysis automatically. Use the button to rerun "
        "the selected video after changing its scoreboard crop."
    )

    if st.button(
        "🚀 Analyze Match",
        type="primary",
        width="stretch",
    ) or not has_cached_results_for_upload:

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            st.error("Scoreboard crop must have positive width and height.")
            st.stop()

        scoreboard_config.update({
            "x1": int(crop_x1), "y1": int(crop_y1),
            "x2": int(crop_x2), "y2": int(crop_y2),
            "reference_width": int(scoreboard_config.get("reference_width", 1280)),
            "reference_height": int(scoreboard_config.get("reference_height", 720)),
        })
        SCOREBOARD_CONFIG.write_text(json.dumps(scoreboard_config, indent=4) + "\n")

        # ----------------------------------------------------
        # DIRECTORIES
        # ----------------------------------------------------

        INPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_DIR / "tracking"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_DIR / "events"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        GOAL_CLIPS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )


        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        status = st.empty()

        progress = st.progress(
            0
        )

        log_placeholder = st.empty()


        # ----------------------------------------------------
        # SAVE VIDEO
        # ----------------------------------------------------

        status.info(
            "Preparing uploaded video..."
        )

        try:

            with open(
                VIDEO_PATH,
                "wb",
            ) as file:

                file.write(
                    uploaded_video.getbuffer()
                )

        except Exception as exc:

            st.error(
                "Could not save uploaded video."
            )

            st.exception(
                exc
            )

            st.stop()


        # ----------------------------------------------------
        # VALIDATE VIDEO
        # ----------------------------------------------------

        valid, video_info = validate_video(
            VIDEO_PATH
        )

        if not valid:

            st.error(
                "Uploaded video validation failed."
            )

            st.code(
                str(video_info)
            )

            st.stop()


        st.success(
            "Uploaded video successfully validated."
        )

        file_size_mb = (
            VIDEO_PATH.stat().st_size
            / (1024 * 1024)
        )

        st.write(
            f"""
**Resolution:** {video_info["width"]} × {video_info["height"]}

**FPS:** {video_info["fps"]:.2f}

**Frames:** {video_info["frames"]:,}

**Duration:** {video_info["duration"]:.2f} seconds

**File size:** {file_size_mb:.2f} MB
"""
        )


        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if not MODEL_PATH.exists():

            st.error(
                "YOLO model was not found."
            )

            st.code(
                str(MODEL_PATH)
            )

            st.stop()


        st.success(
            f"YOLO model found: "
            f"{MODEL_PATH.name}"
        )


        # ----------------------------------------------------
        # CLEAN OLD OUTPUTS
        # ----------------------------------------------------

        clean_previous_outputs()


        # ====================================================
        # PIPELINE
        #
        # IMPORTANT:
        #
        # 1. YOLO
        # 2. Trajectory
        # 3. Movement candidates
        # 4. Movement grouping
        # 5. OCR
        # 6. Score change detection
        # 7. Goal matching
        # 8. Finalize goals
        # 9. Goal clips
        #
        # This order follows the actual file dependencies.
        # ====================================================

        TOTAL_STEPS = 9


        # ====================================================
        # STEP 1
        # YOLO BALL + PLAYER DETECTION
        # ====================================================

        execute_step(
            progress,
            status,
            log_placeholder,
            1,
            TOTAL_STEPS,
            "Detecting football and players with YOLO...",
            PROJECT_ROOT
            / "src"
            / "tracking"
            / "ball_detector.py",
        )

        require_output(
            BALL_DETECTIONS,
            "YOLO ball detections CSV",
        )

        try:

            ball_df = pd.read_csv(
                BALL_DETECTIONS
            )

            st.info(
                f"YOLO generated "
                f"{len(ball_df):,} ball detections."
            )

        except Exception as exc:

            st.error(
                "Ball detection CSV could not be read."
            )

            st.exception(
                exc
            )

            st.stop()


        # ====================================================
        # STEP 2
        # BALL TRAJECTORY
        # ====================================================

        execute_step(
            progress,
            status,
            log_placeholder,
            2,
            TOTAL_STEPS,
            "Building ball trajectory...",
            PROJECT_ROOT
            / "src"
            / "tracking"
            / "build_ball_trajectory.py",
        )

        require_output(
            BALL_TRAJECTORY,
            "Ball trajectory CSV",
        )


        # ====================================================
        # STEP 3
        # BALL MOVEMENT CANDIDATES
        # ====================================================

        execute_step(
            progress,
            status,
            log_placeholder,
            3,
            TOTAL_STEPS,
            "Detecting strong ball movements...",
            PROJECT_ROOT
            / "src"
            / "events"
            / "ball_movement_detector.py",
        )

        require_output(
            MOVEMENT_CANDIDATES,
            "Ball movement candidates CSV",
        )


        # ====================================================
        # STEP 4
        # GROUP MOVEMENT EVENTS
        # ====================================================

        execute_step(
            progress,
            status,
            log_placeholder,
            4,
            TOTAL_STEPS,
            "Grouping ball movements into events...",
            PROJECT_ROOT
            / "src"
            / "events"
            / "group_movement_events.py",
        )

        require_output(
            MOVEMENT_EVENTS,
            "Movement events CSV",
        )


        # ====================================================
        # STEP 5
        # SCOREBOARD OCR
        # ====================================================

        execute_step(
            progress,
            status,
            log_placeholder,
            5,
            TOTAL_STEPS,
            "Reading scoreboard with RapidOCR...",
            PROJECT_ROOT
            / "src"
            / "ocr"
            / "video_scoreboard_reader.py",
        )

        require_output(
            SCOREBOARD_OUTPUT,
            "Scoreboard OCR output",
        )


        # ====================================================
        # STEP 6
        # SCORE CHANGE DETECTION
        # ====================================================

        execute_step(
            progress,
            status,
            log_placeholder,
            6,
            TOTAL_STEPS,
            "Detecting confirmed score changes...",
            PROJECT_ROOT
            / "src"
            / "events"
            / "score_change_detector.py",
        )

        require_output(
            GOAL_CANDIDATES,
            "Goal candidate JSON",
        )


        # ====================================================
        # STEP 7
        # GOAL EVENT MATCHING
        # ====================================================

        execute_step(
            progress,
            status,
            log_placeholder,
            7,
            TOTAL_STEPS,
            "Matching score changes with ball movement...",
            PROJECT_ROOT
            / "src"
            / "events"
            / "goal_event_matcher.py",
        )

        require_output(
            GOAL_EVENTS,
            "Goal events CSV",
        )


        # ====================================================
        # STEP 8
        # FINALIZE GOAL EVENTS
        # ====================================================

        execute_step(
            progress,
            status,
            log_placeholder,
            8,
            TOTAL_STEPS,
            "Finalizing goal events...",
            PROJECT_ROOT
            / "src"
            / "events"
            / "finalize_goal_save_events.py",
        )

        require_output(
            FINAL_GOAL_EVENTS,
            "Final goal events CSV",
        )


        # ====================================================
        # STEP 9
        # GOAL CLIPS
        # ====================================================

        execute_step(
            progress,
            status,
            log_placeholder,
            9,
            TOTAL_STEPS,
            "Extracting goal highlight clips...",
            PROJECT_ROOT
            / "src"
            / "events"
            / "extract_goal_clips.py",
        )

        st.session_state["analyzed_video_sha256"] = uploaded_video_sha256
        st.session_state["analyzed_video_name"] = uploaded_video.name
        ANALYSIS_MANIFEST.write_text(
            json.dumps(
                {
                    "video_sha256": uploaded_video_sha256,
                    "video_name": uploaded_video.name,
                    "video_info": video_info,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        progress.progress(
            1.0
        )

        status.success(
            "✅ Analysis complete!"
        )

        log_placeholder.empty()


        # ----------------------------------------------------
        # PIPELINE SUMMARY
        # ----------------------------------------------------

        st.success(
            "Football Vision pipeline completed successfully."
        )

        col1, col2, col3, col4 = st.columns(4)

        try:

            ball_count = len(
                pd.read_csv(
                    BALL_DETECTIONS
                )
            )

        except Exception:

            ball_count = 0

        try:

            trajectory_count = len(
                pd.read_csv(
                    BALL_TRAJECTORY
                )
            )

        except Exception:

            trajectory_count = 0

        try:

            movement_count = len(
                pd.read_csv(
                    MOVEMENT_EVENTS
                )
            )

        except Exception:

            movement_count = 0

        goals_now = load_goal_events()

        col1.metric(
            "Ball Detections",
            f"{ball_count:,}",
        )

        col2.metric(
            "Trajectory Rows",
            f"{trajectory_count:,}",
        )

        col3.metric(
            "Movement Events",
            f"{movement_count:,}",
        )

        col4.metric(
            "Goals",
            f"{len(goals_now):,}",
        )


# ============================================================
# PER-UPLOAD RESULTS GUARD
# ============================================================

if uploaded_video is None:
    st.info("Upload a match video to analyze it and see its results.")
    st.stop()

results_match_upload = (
    st.session_state.get("analyzed_video_sha256") == uploaded_video_sha256
)
if not results_match_upload and ANALYSIS_MANIFEST.is_file():
    try:
        manifest = json.loads(ANALYSIS_MANIFEST.read_text(encoding="utf-8"))
        results_match_upload = (
            manifest.get("video_sha256") == uploaded_video_sha256
            and FINAL_GOAL_EVENTS.is_file()
            and SCOREBOARD_OUTPUT.is_file()
            and SCORE_CHANGES_OUTPUT.is_file()
        )
        if results_match_upload:
            st.session_state["analyzed_video_sha256"] = uploaded_video_sha256
            st.session_state["analyzed_video_name"] = manifest.get(
                "video_name", uploaded_video.name,
            )
    except (OSError, json.JSONDecodeError):
        results_match_upload = False

if not results_match_upload:
    st.info(
        "This upload has not been analyzed yet. Click **Analyze Match** "
        "to generate its goal highlights, scoreboard OCR, team names, and score changes."
    )
    st.stop()

st.caption(
    "Results for uploaded video: "
    f"**{st.session_state.get('analyzed_video_name', uploaded_video.name)}**"
)


# ============================================================
# RESULTS
# ============================================================

st.divider()

st.header(
    "2. Goal Highlights"
)

goals = load_goal_events()


# ============================================================
# GOAL RESULTS
# ============================================================

if goals.empty:

    st.info(
        "No confirmed goal events detected."
    )

else:

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Goals Detected",
        len(goals),
    )

    if "scoring_team" in goals.columns:

        teams = (
            goals["scoring_team"]
            .dropna()
            .value_counts()
        )

        col2.metric(
            "Scoring Teams",
            len(teams),
        )

    else:

        col2.metric(
            "Scoring Teams",
            "-",
        )

    if "confidence" in goals.columns:

        avg_confidence = (
            pd.to_numeric(
                goals["confidence"],
                errors="coerce",
            )
            .mean()
        )

        if pd.notna(avg_confidence):

            col3.metric(
                "Average Confidence",
                f"{avg_confidence:.2f}",
            )

        else:

            col3.metric(
                "Average Confidence",
                "-",
            )

    else:

        col3.metric(
            "Average Confidence",
            "-",
        )


    st.divider()


    # --------------------------------------------------------
    # FIND GENERATED CLIPS
    # --------------------------------------------------------

    clip_files = sorted(
        GOAL_CLIPS_DIR.glob(
            "goal_*.mp4"
        )
    )


    # --------------------------------------------------------
    # DISPLAY GOALS
    # --------------------------------------------------------

    for goal_number, (_, row) in enumerate(
        goals.sort_values(
            "event_time"
        ).iterrows(),
        start=1,
    ):

        event_time = float(
            row.get(
                "event_time",
                0,
            )
        )

        scoring_team = row.get(
            "scoring_team",
            "?",
        )

        clock = row.get(
            "clock",
            "?",
        )

        details = row.get(
            "details",
            "",
        )

        confidence = row.get(
            "confidence",
            0,
        )

        st.subheader(
            f"⚽ Goal {goal_number}"
        )

        col1, col2 = st.columns(
            [2, 1]
        )


        # ----------------------------------------------------
        # VIDEO
        # ----------------------------------------------------

        with col1:

            if (
                goal_number <=
                len(clip_files)
            ):

                clip_file = (
                    clip_files[
                        goal_number - 1
                    ]
                )

                st.video(
                    str(clip_file)
                )

                with open(
                    clip_file,
                    "rb",
                ) as video:

                    st.download_button(
                        "⬇ Download Goal Clip",
                        data=video,
                        file_name=clip_file.name,
                        mime="video/mp4",
                        key=(
                            f"goal_download_"
                            f"{goal_number}"
                        ),
                    )

            else:

                st.warning(
                    "Goal clip was not generated."
                )


        # ----------------------------------------------------
        # GOAL DETAILS
        # ----------------------------------------------------

        with col2:

            st.markdown(
                f"""
**Scoring Team:** {scoring_team}

**Match Clock:** {clock}

**Event Time:** {event_time:.2f}s

**Score Change:** {details}

**Confidence:** {float(confidence):.2f}
"""
            )

            if "score_change_time" in row:

                try:

                    score_change_time = float(
                        row[
                            "score_change_time"
                        ]
                    )

                    st.write(
                        f"Scoreboard change: "
                        f"{score_change_time:.2f}s"
                    )

                except Exception:
                    pass


        st.divider()


# ============================================================
# SCORE CHANGES
# ============================================================

st.header(
    "3. Score Changes"
)

if SCORE_CHANGES_OUTPUT.exists():

    try:
        score_changes = pd.read_csv(SCORE_CHANGES_OUTPUT)
    except Exception:
        score_changes = pd.DataFrame()

else:
    score_changes = pd.DataFrame()

if score_changes.empty:

    st.info(
        "No confirmed score changes detected in the scoreboard OCR."
    )

else:

    columns = [
        "video_time",
        "clock",
        "home_team",
        "previous_home_score",
        "previous_away_score",
        "new_home_score",
        "new_away_score",
        "away_team",
        "change_type",
    ]

    available_columns = [
        column
        for column in columns
        if column in score_changes.columns
    ]

    score_table = score_changes[available_columns].copy()

    st.dataframe(
        score_table,
        width="stretch",
        hide_index=True,
    )

    score_csv = score_table.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download Score Changes",
        data=score_csv,
        file_name="score_changes.csv",
        mime="text/csv",
    )


# ============================================================
# SCOREBOARD OCR
# ============================================================

st.header(
    "4. Scoreboard OCR"
)

ocr_readings = load_ocr()


if not ocr_readings:

    st.info(
        "No scoreboard OCR readings available."
    )

else:

    ocr_df = pd.DataFrame(
        ocr_readings
    )

    valid_score_count = 0
    if {"home_score", "away_score"}.issubset(ocr_df.columns):
        valid_score_count = int(
            ocr_df["home_score"].notna().astype(bool)
            .mul(ocr_df["away_score"].notna().astype(bool)).sum()
        )

    st.write(
        f"Total OCR readings: "
        f"**{len(ocr_df)}**"
    )

    if valid_score_count == 0:
        st.error(
            "Scoreboard OCR found no complete score readings. Goal detection "
            "cannot be confirmed for this run. Check the scoreboard crop "
            "settings in the sidebar and analyze the video again."
        )
    else:
        st.caption(
            f"Valid scoreboard scores: {valid_score_count:,} / {len(ocr_df):,}"
        )


    # --------------------------------------------------------
    # LATEST VALID SCOREBOARD
    # --------------------------------------------------------

    valid = ocr_df.copy()

    if "clock" in valid.columns:

        valid = valid[
            valid["clock"].notna()
        ]

    if not valid.empty:

        latest = valid.iloc[-1]

        st.subheader(
            "Latest Scoreboard Reading"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Clock",
            str(
                latest.get(
                    "clock",
                    "-",
                )
            ),
        )

        col2.metric(
            "Score",
            (
                f"{latest.get('home_score', '-')}"
                f" - "
                f"{latest.get('away_score', '-')}"
            ),
        )

        col3.metric(
            "Teams",
            (
                f"{latest.get('home_team', '-')}"
                f" vs "
                f"{latest.get('away_team', '-')}"
            ),
        )

        st.write(
            format_scoreboard(
                latest
            )
        )


    # --------------------------------------------------------
    # OCR TIMELINE
    # --------------------------------------------------------

    st.subheader(
        "OCR Timeline"
    )

    st.dataframe(
        ocr_df,
        width="stretch",
        hide_index=True,
    )


    # --------------------------------------------------------
    # OCR DOWNLOAD
    # --------------------------------------------------------

    ocr_csv = ocr_df.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download OCR Results",
        data=ocr_csv,
        file_name="scoreboard_readings.csv",
        mime="text/csv",
    )


# ============================================================
# PIPELINE
# ============================================================

st.divider()

st.header(
    "5. Pipeline"
)

st.code(
    """
Uploaded Match Video
        │
        ▼
YOLO Ball + Player Detection
        │
        ▼
Ball Detections CSV
        │
        ▼
Ball Trajectory
        │
        ▼
Ball Movement Candidates
        │
        ▼
Grouped Movement Events
        │
        ├─────────────────────┐
        │                     │
        ▼                     ▼
Scoreboard Crop          Ball Movement
        │                     │
        ▼                     │
RapidOCR                     │
        │                     │
        ▼                     │
Scoreboard Parser            │
        │                     │
        ▼                     │
Score Change Detection       │
        │                     │
        └──────────┬──────────┘
                   │
                   ▼
          Goal Event Matching
                   │
                   ▼
          Final Goal Events
                   │
                   ▼
          Goal Highlight Clips
                   │
                   ▼
          Streamlit Results
""",
    language="text",
)


# ============================================================
# DEBUG / ENVIRONMENT INFORMATION
# ============================================================

with st.expander(
    "Environment / Debug Information"
):

    st.write(
        "Python executable:"
    )

    st.code(
        sys.executable
    )

    st.write(
        "Project root:"
    )

    st.code(
        str(PROJECT_ROOT)
    )

    st.write(
        "YOLO model:"
    )

    st.code(
        str(MODEL_PATH)
    )

    st.write(
        "Uploaded video:"
    )

    st.code(
        str(VIDEO_PATH)
    )

    st.write(
        "Goal clips directory:"
    )

    st.code(
        str(GOAL_CLIPS_DIR)
    )
