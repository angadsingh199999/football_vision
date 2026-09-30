import os
import pandas as pd


OUTPUT_CSV = "outputs/highlights/final_highlight_index.csv"


def get_clip(path):
    if os.path.exists(path):
        return path
    return None


def main():

    print("=" * 70)
    print("CREATING FINAL HIGHLIGHT INDEX")
    print("=" * 70)

    rows = []

    # ==========================================================
    # GOALS
    # ==========================================================

    goals = pd.read_csv(
        "outputs/events/goal_events.csv"
    )

    for index, row in goals.iterrows():

        goal_id = index + 1

        clip = get_clip(
            f"outputs/highlights/key_moments/goal_{goal_id}.mp4"
        )

        rows.append({
            "event_type": "GOAL",
            "event_id": goal_id,
            "time": float(row["goal_time"]),
            "start_time": float(row["clip_start"]),
            "end_time": float(row["clip_end"]),
            "confidence": 1.0,
            "clip": clip
        })

    # ==========================================================
    # SHOTS
    # ==========================================================

    shots = pd.read_csv(
        "outputs/events/final_shot_events.csv"
    )

    for _, row in shots.iterrows():

        shot_id = int(row["shot_id"])

        clip = get_clip(
            f"outputs/highlights/shots/shot_{shot_id}.mp4"
        )

        rows.append({
            "event_type": "SHOT",
            "event_id": shot_id,
            "time": float(row["peak_time"]),
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "confidence": float(row["peak_confidence"]),
            "clip": clip
        })

    # ==========================================================
    # SAVES
    # ==========================================================

    saves = pd.read_csv(
        "outputs/events/final_save_events.csv"
    )

    save_clip_number = 1

    for _, row in saves.iterrows():

        shot_id = int(row["shot_id"])
        shot_time = float(row["shot_time"])

        clip = get_clip(
            f"outputs/highlights/saves/save_{save_clip_number}.mp4"
        )

        rows.append({
            "event_type": "SAVE",
            "event_id": shot_id,
            "time": shot_time,
            "start_time": shot_time - 5.0,
            "end_time": shot_time + 5.0,
            "confidence": float(row["save_confidence"]),
            "clip": clip
        })

        save_clip_number += 1

    # ==========================================================
    # DANGEROUS ATTACKS
    # ==========================================================

    attacks = pd.read_csv(
        "outputs/events/final_dangerous_attacks.csv"
    )

    for _, row in attacks.iterrows():

        attack_id = int(row["final_attack_id"])

        clip = get_clip(
            f"outputs/highlights/attacks/attack_{attack_id}.mp4"
        )

        rows.append({
            "event_type": "DANGEROUS_ATTACK",
            "event_id": attack_id,
            "time": float(row["peak_time"]),
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "confidence": float(row["attack_score"]),
            "clip": clip
        })

    # ==========================================================
    # CORNERS
    # ==========================================================

    corners = pd.read_csv(
        "outputs/events/final_corner_events.csv"
    )

    for _, row in corners.iterrows():

        corner_id = int(row["final_corner_id"])
        peak_time = float(row["peak_time"])

        clip = get_clip(
            f"outputs/highlights/corners/corner_{corner_id}.mp4"
        )

        rows.append({
            "event_type": "CORNER",
            "event_id": corner_id,
            "time": peak_time,
            "start_time": max(0.0, peak_time - 5.0),
            "end_time": peak_time + 5.0,
            "confidence": float(row["corner_score"]),
            "clip": clip
        })

    # ==========================================================
    # FOUL CANDIDATES
    # ==========================================================

    fouls = pd.read_csv(
        "outputs/events/filtered_foul_candidates.csv"
    )

    for _, row in fouls.iterrows():

        foul_id = int(row["final_foul_id"])
        peak_time = float(row["peak_time"])

        clip = get_clip(
            f"outputs/highlights/fouls/foul_{foul_id}.mp4"
        )

        rows.append({
            "event_type": "FOUL_CANDIDATE",
            "event_id": foul_id,
            "time": peak_time,
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "confidence": float(row["foul_score"]),
            "clip": clip
        })

    # ==========================================================
    # DATAFRAME
    # ==========================================================

    result = pd.DataFrame(rows)

    if result.empty:
        print("ERROR: No events found.")
        return

    # Sort chronologically
    result = result.sort_values(
        "time"
    ).reset_index(drop=True)

    # Master event ID
    result.insert(
        0,
        "master_event_id",
        range(1, len(result) + 1)
    )

    # Check clips
    result["clip_exists"] = result["clip"].apply(
        lambda x: (
            isinstance(x, str)
            and os.path.exists(x)
        )
    )

    # ==========================================================
    # SAVE
    # ==========================================================

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ==========================================================
    # OUTPUT
    # ==========================================================

    print()
    print(f"Total events: {len(result)}")

    print(
        f"Events with clips: "
        f"{int(result['clip_exists'].sum())}"
    )

    print(
        f"Events missing clips: "
        f"{int((~result['clip_exists']).sum())}"
    )

    print()

    print(
        result[
            [
                "master_event_id",
                "event_type",
                "event_id",
                "time",
                "confidence",
                "clip",
                "clip_exists"
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

    print("=" * 70)


if __name__ == "__main__":
    main()