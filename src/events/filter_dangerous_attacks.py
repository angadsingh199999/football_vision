import pandas as pd


INPUT_CSV = "outputs/events/dangerous_attack_candidates.csv"
OUTPUT_CSV = "outputs/events/final_dangerous_attacks.csv"

MIN_SPEED = 100
MIN_DETECTIONS = 2
MIN_NEARBY_PLAYERS = 20

# The clip extractor uses:
#   peak_time - 5 seconds
#   peak_time + 5 seconds
#
# Therefore two attacks whose peaks are less than 10 seconds
# apart would produce overlapping clips.
CLIP_HALF_DURATION = 5.0


def main():

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        print("No attack candidates found.")
        return

    # ========================================================
    # BASIC FILTERING
    # ========================================================

    filtered = df[
        (df["peak_speed"] >= MIN_SPEED) &
        (df["detections"] >= MIN_DETECTIONS) &
        (df["max_nearby_players"] >= MIN_NEARBY_PLAYERS)
    ].copy()

    print("=" * 60)
    print("FINAL DANGEROUS ATTACKS")
    print("=" * 60)

    print(
        f"Candidates before filtering: {len(df)}"
    )

    print(
        f"Candidates after basic filtering: {len(filtered)}"
    )

    if filtered.empty:

        columns = [
            "final_attack_id",
            "attack_id",
            "start_time",
            "end_time",
            "peak_time",
            "peak_speed",
            "attacking_goal",
            "max_nearby_players",
            "detections",
            "attack_score"
        ]

        pd.DataFrame(
            columns=columns
        ).to_csv(
            OUTPUT_CSV,
            index=False
        )

        print(
            f"Saved: {OUTPUT_CSV}"
        )

        print("=" * 60)

        return

    # ========================================================
    # REMOVE OVERLAPPING ATTACK CLIPS
    # ========================================================
    #
    # We process strongest attacks first.
    #
    # If a weaker attack would produce a clip overlapping
    # an already accepted stronger attack, we reject it.
    #
    # This directly prevents duplicate highlight clips.
    # ========================================================

    ranked = filtered.sort_values(
        [
            "attack_score",
            "peak_speed"
        ],
        ascending=[
            False,
            False
        ]
    ).reset_index(
        drop=True
    )

    accepted = []
    rejected = []

    for _, row in ranked.iterrows():

        peak_time = float(
            row["peak_time"]
        )

        candidate_start = (
            peak_time
            - CLIP_HALF_DURATION
        )

        candidate_end = (
            peak_time
            + CLIP_HALF_DURATION
        )

        overlaps = False
        overlapping_attack_id = None

        for accepted_row in accepted:

            accepted_peak = float(
                accepted_row["peak_time"]
            )

            accepted_start = (
                accepted_peak
                - CLIP_HALF_DURATION
            )

            accepted_end = (
                accepted_peak
                + CLIP_HALF_DURATION
            )

            # Interval overlap.
            if (
                candidate_start < accepted_end
                and
                candidate_end > accepted_start
            ):

                overlaps = True

                overlapping_attack_id = int(
                    accepted_row["attack_id"]
                )

                break

        if overlaps:

            rejected.append({
                "attack_id": int(
                    row["attack_id"]
                ),
                "peak_time": peak_time,
                "reason": (
                    "overlaps_stronger_attack_"
                    + str(overlapping_attack_id)
                )
            })

        else:

            accepted.append(row.to_dict())

    # ========================================================
    # CREATE FINAL DATAFRAME
    # ========================================================

    final = pd.DataFrame(
        accepted
    )

    if not final.empty:

        # Chronological ordering is much easier to understand
        # when reviewing generated clips.
        final = final.sort_values(
            "peak_time"
        ).reset_index(
            drop=True
        )

        final["final_attack_id"] = range(
            1,
            len(final) + 1
        )

        columns = [
            "final_attack_id",
            "attack_id",
            "start_time",
            "end_time",
            "peak_time",
            "peak_speed",
            "attacking_goal",
            "max_nearby_players",
            "detections",
            "attack_score"
        ]

        final = final[
            columns
        ]

    else:

        final = pd.DataFrame(
            columns=[
                "final_attack_id",
                "attack_id",
                "start_time",
                "end_time",
                "peak_time",
                "peak_speed",
                "attacking_goal",
                "max_nearby_players",
                "detections",
                "attack_score"
            ]
        )

    # ========================================================
    # SAVE
    # ========================================================

    final.to_csv(
        OUTPUT_CSV,
        index=False
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        f"Overlapping attacks removed: "
        f"{len(rejected)}"
    )

    print(
        f"Final attacks: {len(final)}"
    )

    print()

    if not final.empty:

        print(
            final.to_string(
                index=False
            )
        )

    print()

    if rejected:

        print("REJECTED OVERLAPPING ATTACKS")
        print("-" * 60)

        for item in rejected:

            print(
                f"Attack {item['attack_id']} "
                f"at {item['peak_time']:.2f}s: "
                f"{item['reason']}"
            )

        print()

    print(
        f"Saved: {OUTPUT_CSV}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()