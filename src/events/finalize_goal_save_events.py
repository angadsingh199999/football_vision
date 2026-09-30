from pathlib import Path
import pandas as pd

GOAL_FILE = Path("outputs/events/goal_events.csv")
OUTPUT_FILE = Path("outputs/events/final_goal_events.csv")

print("=" * 70)
print("FINAL GOAL EVENT RESOLUTION")
print("=" * 70)

if not GOAL_FILE.exists():
    raise FileNotFoundError(
        f"Goal events file not found: {GOAL_FILE}"
    )

goals = pd.read_csv(GOAL_FILE)

if goals.empty:
    print("No goal events detected.")
    final_events = pd.DataFrame(
        columns=[
            "event_id",
            "event_type",
            "event_time",
            "start_time",
            "end_time",
            "confidence",
            "scoring_team",
            "clock",
            "details",
        ]
    )
else:
    final_events = []

    for event_id, (_, goal) in enumerate(
        goals.sort_values("goal_time").iterrows(),
        start=1
    ):
        goal_time = float(goal["goal_time"])

        final_events.append({
            "event_id": event_id,
            "event_type": "Goal",
            "event_time": goal_time,
            "start_time": float(goal["clip_start"]),
            "end_time": float(goal["clip_end"]),
            "confidence": float(goal["confidence"]),
            "scoring_team": goal["scoring_team"],
            "clock": goal["clock"],
            "details": (
                f'{goal["previous_score"]} -> '
                f'{goal["new_score"]}'
            ),
        })

    final_events = pd.DataFrame(final_events)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
final_events.to_csv(OUTPUT_FILE, index=False)

print()
print("FINAL GOALS")
print("-" * 70)

if final_events.empty:
    print("No goals found.")
else:
    print(final_events.to_string(index=False))

print()
print(f"Goals : {len(final_events)}")
print("Saves : 0")
print(f"Saved : {OUTPUT_FILE}")
print("=" * 70)