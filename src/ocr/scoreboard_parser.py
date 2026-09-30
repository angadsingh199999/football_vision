import re


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    """
    Normalize OCR text.
    """

    if text is None:
        return ""

    text = str(text)

    text = (
        text
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )

    return text


def normalize_ocr_text(text):
    """
    Normalize common OCR mistakes.
    """

    text = clean_text(text)

    # Common OCR substitutions.
    # We only use these when dealing with score-like text.
    replacements = {
        "—": "-",
        "–": "-",
        "_": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# ============================================================
# CLOCK
# ============================================================

def find_clock(texts):

    for text in texts:

        text = normalize_ocr_text(text)

        match = re.search(
            r"\b(\d{1,2}):(\d{2})\b",
            text
        )

        if match:

            minutes = int(match.group(1))
            seconds = int(match.group(2))

            if minutes <= 120 and seconds < 60:

                return (
                    f"{minutes:02d}:"
                    f"{seconds:02d}"
                )

    # Some scoreboards are returned as separate plain number tokens, e.g.
    # (..., "LEG", "29", "17") for 29:17. The clock is generally the final
    # adjacent numeric pair after team labels; scan from the end to avoid
    # interpreting the score pair as a clock.
    for left, right in reversed(list(zip(texts, texts[1:]))):
        left = normalize_ocr_text(left).strip()
        right = normalize_ocr_text(right).strip()
        left_digits = re.sub(r"\D", "", left)
        right_digits = re.sub(r"\D", "", right)
        if (
            re.fullmatch(r"\d{1,3}:?", left)
            and re.fullmatch(r":?\d{2}", right)
        ):
            minutes = int(left_digits)
            seconds = int(right_digits)
            if minutes <= 120 and seconds < 60:
                return f"{minutes:02d}:{seconds:02d}"

    return None


# ============================================================
# SCORE PAIR
# ============================================================

def find_score_pair(texts):

    score_texts = list(texts)

    # Remove a trailing clock represented as two separate number tokens
    # (e.g. ... "LEG", "29", "15") before collecting standalone score digits.
    # Require preceding text so a bare pair like ["0", "1"] remains a score.
    if len(score_texts) >= 4:
        left = normalize_ocr_text(score_texts[-2]).strip()
        right = normalize_ocr_text(score_texts[-1]).strip()
        has_label_before = any(
            re.search(r"[A-Za-z]", normalize_ocr_text(token))
            for token in score_texts[:-2]
        )
        if (
            has_label_before
            and re.fullmatch(r"\d{1,3}:?", left)
            and re.fullmatch(r":?\d{2}", right)
            and int(re.sub(r"\D", "", left)) <= 120
            and int(re.sub(r"\D", "", right)) < 60
        ):
            score_texts = score_texts[:-2]

    # --------------------------------------------------------
    # First pass:
    # Look for explicit forms such as:
    #
    # 0-0
    # 1-0
    # 2 - 1
    # 00
    # --------------------------------------------------------

    for text in score_texts:

        text = normalize_ocr_text(text)

        # 0-0 / 1-0 / 2-1
        matches = re.findall(
            r"(?<!\d)(\d{1,2})\s*-\s*(\d{1,2})(?!\d)",
            text
        )

        for home, away in matches:

            home = int(home)
            away = int(away)

            if home <= 20 and away <= 20:

                return home, away


    # --------------------------------------------------------
    # Second pass:
    # Find standalone numeric values.
    #
    # Example:
    # 21:11
    # NAP
    # 0
    # 0
    # BAR
    # --------------------------------------------------------

    numbers = []

    for text in score_texts:

        text = normalize_ocr_text(text)

        # Remove clock first so 21:11 doesn't
        # become 21 and 11.
        text_without_clock = re.sub(
            r"\b\d{1,2}:\d{2}\b",
            " ",
            text
        )

        matches = re.findall(
            r"(?<![\w])(\d{1,2})(?![\w])",
            text_without_clock
        )

        for value in matches:

            number = int(value)

            # Football scores are normally small.
            if 0 <= number <= 20:

                numbers.append(number)


    if len(numbers) >= 2:

        return (
            numbers[0],
            numbers[1]
        )


    # --------------------------------------------------------
    # Third pass:
    # OCR may produce "00", "10", "21".
    #
    # Try splitting two-digit score strings.
    # --------------------------------------------------------

    for text in score_texts:

        text = normalize_ocr_text(text)

        text = re.sub(
            r"\b\d{1,2}:\d{2}\b",
            " ",
            text
        )

        compact_values = re.findall(
            r"\b\d{2}\b",
            text
        )

        for value in compact_values:

            home = int(value[0])
            away = int(value[1])

            if home <= 9 and away <= 9:

                return home, away


    return None


# ============================================================
# TEAM NAMES
# ============================================================

def find_team_names(texts):
    tokens = [normalize_ocr_text(text).strip() for text in texts]

    def team_token(index):
        if index < 0 or index >= len(tokens):
            return None
        value = tokens[index]
        if re.fullmatch(r"[A-Za-z]{2,}", value):
            return value
        return None

    all_labels = [team_token(i) for i in range(len(tokens)) if team_token(i)]

    # Prefer labels immediately surrounding the score. This prevents nearby
    # graphics (often misread as short words) from replacing a real team.
    for index, token in enumerate(tokens):
        match = re.search(r"(?<!\d)(\d{1,2})\s*-\s*(\d{1,2})(?!\d)", token)
        if match:
            home = next((team_token(i) for i in range(index - 1, -1, -1) if team_token(i)), None)
            away = next((team_token(i) for i in range(index + 1, len(tokens)) if team_token(i)), None)
            if home is None and all_labels:
                home = all_labels[0]
            if away is None or away == home:
                away = next((label for label in all_labels if label != home), None)
            return home, away

    # OCR sometimes emits the digits and hyphen as separate tokens. Locate
    # the first adjacent low-valued numeric pair and use its neighboring text.
    for index in range(len(tokens) - 1):
        left = tokens[index].strip()
        right = tokens[index + 1].strip()
        if re.fullmatch(r"\d{1,2}", left) and re.fullmatch(r"-?\d{1,2}", right):
            if int(left) <= 20 and int(right.lstrip("-")) <= 20:
                home = next((team_token(i) for i in range(index - 1, -1, -1) if team_token(i)), None)
                away = next((team_token(i) for i in range(index + 2, len(tokens)) if team_token(i)), None)
                if home is None and all_labels:
                    home = all_labels[0]
                if away is None or away == home:
                    away = next((label for label in all_labels if label != home), None)
                return home, away

    labels = [team_token(i) for i in range(len(tokens)) if team_token(i)]
    if len(labels) >= 2:
        return labels[0], labels[1]
    if labels:
        return labels[0], None
    return None, None


# ============================================================
# MAIN PARSER
# ============================================================

def parse_scoreboard(texts):

    if texts is None:
        texts = []

    texts = [
        clean_text(text)
        for text in texts
        if clean_text(text)
    ]


    result = {
        "clock": None,
        "home_team": None,
        "home_score": None,
        "away_score": None,
        "away_team": None,
    }


    if not texts:
        return result


    # --------------------------------------------------------
    # Clock
    # --------------------------------------------------------

    # RapidOCR can split a clock such as 05:07 into adjacent tokens "05"
    # and ":07". Also test the joined text while retaining the original
    # tokens for score and team parsing.
    joined_texts = list(texts)
    if len(texts) > 1:
        joined_texts.append(" ".join(texts))
        joined_texts.extend(
            f"{left}{right}"
            for left, right in zip(texts, texts[1:])
        )
    result["clock"] = find_clock(joined_texts)


    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    score = find_score_pair(
        texts
    )

    if score is not None:

        result["home_score"] = score[0]
        result["away_score"] = score[1]


    # --------------------------------------------------------
    # Teams
    # --------------------------------------------------------

    home_team, away_team = find_team_names(
        texts
    )

    result["home_team"] = home_team
    result["away_team"] = away_team


    return result
