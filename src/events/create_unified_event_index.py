import os
import pandas as pd


OUTPUT_CSV = "outputs/events/unified_event_index.csv"


def load_csv(path):
    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def main():

    print("=" * 60)
    print("CREATING UNIFIED EVENT INDEX")
    print("=" * 60)

    events = []

    # ==================================================
    # GOALS
    # ==================================================

    goals = load_csv(
        "outputs/events/goal_events.csv"
    )

    for _, row in goals.iterrows():

        events.append({
            "time": float(row["goal_time"]),
            "event_type": "GOAL",
            "event_id": None,
            "start_time": float(row["clip_start"]),
            "end_time": float(row["clip_end"]),
            "confidence": 1.0,
            "details": (
                f'{row["scoring_team"]} '
                f'{row["previous_score"]} -> '
                f'{row["new_score"]}'
            )
        })

    # ==================================================
    # SHOTS
    # ==================================================

    shots = load_csv(
        "outputs/events/final_shot_events.csv"
    )

    for _, row in shots.iterrows():

        events.append({
            "time": float(row["peak_time"]),
            "event_type": "SHOT",
            "event_id": int(row["shot_id"]),
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "confidence": float(row["peak_confidence"]),
            "details": (
                f'{row["direction"]}, '
                f'speed={float(row["peak_speed"]):.2f}'
            )
        })

    # ==================================================
    # SAVES
    # ==================================================

    saves = load_csv(
        "outputs/events/final_save_events.csv"
    )

    for _, row in saves.iterrows():

        events.append({
            "time": float(row["shot_time"]),
            "event_type": "SAVE",
            "event_id": int(row["shot_id"]),
            "start_time": float(row["shot_time"]) - 5.0,
            "end_time": float(row["shot_time"]) + 5.0,
            "confidence": float(row["save_confidence"]),
            "details": (
                f'{row["direction"]}, '
                f'{row["classification"]}'
            )
        })

    # ==================================================
    # DANGEROUS ATTACKS
    # ==================================================

    attacks = load_csv(
        "outputs/events/final_dangerous_attacks.csv"
    )

    for _, row in attacks.iterrows():

        events.append({
            "time": float(row["peak_time"]),
            "event_type": "DANGEROUS_ATTACK",
            "event_id": int(row["final_attack_id"]),
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "confidence": float(row["attack_score"]),
            "details": (
                f'{row["attacking_goal"]}, '
                f'speed={float(row["peak_speed"]):.2f}'
            )
        })

    # ==================================================
    # CORNERS
    # ==================================================

    corners = load_csv(
        "outputs/events/final_corner_events.csv"
    )

    for _, row in corners.iterrows():

        events.append({
            "time": float(row["peak_time"]),
            "event_type": "CORNER",
            "event_id": int(row["final_corner_id"]),
            "start_time": max(
                0.0,
                float(row["peak_time"]) - 5.0
            ),
            "end_time": float(row["peak_time"]) + 5.0,
            "confidence": float(row["corner_score"]),
            "details": (
                f'{row["boundary"]}, '
                f'nearest_player='
                f'{float(row["nearest_player"]):.2f}'
            )
        })

    # ==================================================
    # FOUL CANDIDATES
    # ==================================================

    fouls = load_csv(
        "outputs/events/filtered_foul_candidates.csv"
    )

    for _, row in fouls.iterrows():

        events.append({
            "time": float(row["peak_time"]),
            "event_type": "FOUL_CANDIDATE",
            "event_id": int(row["final_foul_id"]),
            "start_time": max(
                0.0,
                float(row["peak_time"]) - 5.0
            ),
            "end_time": float(row["peak_time"]) + 5.0,
            "confidence": float(row["foul_score"]),
            "details": (
                f'nearby_players='
                f'{int(row["nearby_players"])}, '
                f'nearest_player='
                f'{float(row["nearest_player"]):.2f}'
            )
        })

    # ==================================================
    # CREATE DATAFRAME
    # ==================================================

    result = pd.DataFrame(events)

    if result.empty:

        print("No events found.")
        return

    # ==================================================
    # SORT CHRONOLOGICALLY
    # ==================================================

    result = result.sort_values(
        "time"
    ).reset_index(drop=True)

    # ==================================================
    # MASTER EVENT ID
    # ==================================================

    result.insert(
        0,
        "master_event_id",
        range(1, len(result) + 1)
    )

    # ==================================================
    # SAVE
    # ==================================================

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ==================================================
    # DISPLAY
    # ==================================================

    print()
    print(
        f"Total unified events: {len(result)}"
    )

    print()

    print(
        result[
            [
                "master_event_id",
                "time",
                "event_type",
                "event_id",
                "start_time",
                "end_time",
                "confidence",
                "details"
            ]
        ].to_string(index=False)
    )

    print()

    print("Event counts:")
    print(
        result["event_type"].value_counts()
    )

    print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()