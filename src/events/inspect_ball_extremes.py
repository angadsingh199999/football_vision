import pandas as pd


TRAJECTORY_CSV = "outputs/tracking/ball_trajectory.csv"


def main():

    df = pd.read_csv(TRAJECTORY_CSV)

    print("=" * 60)
    print("BALL TRAJECTORY EXTREMES")
    print("=" * 60)

    print(f"Total trajectory rows: {len(df)}")
    print()

    print("LEFTMOST BALL POSITIONS")
    print("-" * 60)

    left = df.nsmallest(20, "center_x")

    print(
        left[
            [
                "time",
                "center_x",
                "center_y",
                "speed",
                "confidence"
            ]
        ].to_string(index=False)
    )

    print()

    print("RIGHTMOST BALL POSITIONS")
    print("-" * 60)

    right = df.nlargest(20, "center_x")

    print(
        right[
            [
                "time",
                "center_x",
                "center_y",
                "speed",
                "confidence"
            ]
        ].to_string(index=False)
    )

    print()

    print("TOPMOST BALL POSITIONS")
    print("-" * 60)

    top = df.nsmallest(20, "center_y")

    print(
        top[
            [
                "time",
                "center_x",
                "center_y",
                "speed",
                "confidence"
            ]
        ].to_string(index=False)
    )

    print()

    print("BOTTOMMOST BALL POSITIONS")
    print("-" * 60)

    bottom = df.nlargest(20, "center_y")

    print(
        bottom[
            [
                "time",
                "center_x",
                "center_y",
                "speed",
                "confidence"
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()