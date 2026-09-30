# Football Vision

Football Vision is a Streamlit app that analyzes an uploaded football match video. It reads the broadcast scoreboard, detects confirmed score changes, estimates goal moments, and creates short goal clips.

## Features

- Upload a match video in Streamlit; a new upload starts analysis automatically.
- Detect the ball and players with the included YOLO model at `models/best.pt`.
- Build a ball trajectory and identify movement events.
- Read scoreboard team names, score, and match clock with RapidOCR.
- Show the OCR timeline and confirmed score changes in the app.
- Match score changes to nearby ball movement and create goal highlight clips.
- Keep results associated with the exact uploaded video so a previous run is not shown for a different upload.

## Run the Streamlit app

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a video in the app. A new video is analyzed automatically. To rerun the selected video after changing the scoreboard crop, use **Analyze Match**.

## Scoreboard setup

The default crop covers a common top-left scoreboard layout. The OCR stage scales the crop to the video's resolution and tries a wider top-left crop if the configured crop misses the scoreboard. For other layouts, change the crop in the app's sidebar and rerun the video. Include both team labels, the score, and match clock in the crop.

Scoreboard OCR depends on the overlay being visible and readable. Goal time is localized from nearby ball movement when available; if tracking misses the event, the app estimates a time shortly before the confirmed scoreboard change, so inspect those clips.

## Outputs

Generated files are saved under `outputs/`:

- `scoreboard_readings.json` — OCR readings over the video timeline.
- `score_changes.csv` — confirmed score changes and scoreboard corrections.
- `events/final_goal_events.csv` — detected goals and estimated event times.
- `highlights/goals/` — extracted goal highlight clips.
- `highlights/goal_clip_validation.csv` — clip readability validation report.

The app recreates outputs for each analyzed upload. Videos under `data/input/`, extracted frames, and generated output files are excluded from Git.

## Command-line pipeline

To process a local video without Streamlit, place it at `data/input/match.mp4`, then run:

```bash
source .venv/bin/activate
python run_pipeline.py
```

## Model

The repository includes `models/best.pt`, the YOLO weights used by the pipeline. The model detects ball and player classes; goal decisions also rely on scoreboard changes and ball movement rather than a standalone trained goal classifier.
