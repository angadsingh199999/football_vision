import pandas as pd


INPUT_CSV = "outputs/events/analyzed_save_candidates.csv"
OUTPUT_CSV = "outputs/events/final_save_events.csv"


def classify(row):

    distance = float(row["goal_distance"])
    movement = float(row["movement_after"])
    moved_away = bool(row["moved_away_from_goal"])

    # Strong evidence:
    # ball gets very close to goal and clearly moves away
    if distance <= 40 and moved_away and movement >= 100:
        return "STRONG_SAVE"

    # Moderate evidence
    if distance <= 80 and moved_away and movement >= 50:
        return "POSSIBLE_SAVE"

    return "WEAK_CANDIDATE"


def main():

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        print("No save candidates found.")
        return

    df["classification"] = df.apply(
        classify,
        axis=1
    )

    # Confidence score
    def confidence(row):

        if row["classification"] == "STRONG_SAVE":
            return 0.90

        if row["classification"] == "POSSIBLE_SAVE":
            return 0.65

        return 0.30

    df["save_confidence"] = df.apply(
        confidence,
        axis=1
    )

    # Keep actual useful save events
    result = df[
        df["classification"].isin(
            ["STRONG_SAVE", "POSSIBLE_SAVE"]
        )
    ].copy()

    result = result.sort_values(
        "save_confidence",
        ascending=False
    )

    result.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("=" * 40)
    print("FINAL SAVE EVENTS")
    print("=" * 40)

    print(
        f"Save events: {len(result)}"
    )

    print()

    if not result.empty:
        print(
            result[
                [
                    "shot_id",
                    "shot_time",
                    "direction",
                    "goal_distance",
                    "movement_after",
                    "classification",
                    "save_confidence"
                ]
            ].to_string(index=False)
        )

    print()
    print(f"Saved: {OUTPUT_CSV}")
    print("=" * 40)


if __name__ == "__main__":
    main()