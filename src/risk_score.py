import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


# Thresholds are defined as module-level constants so they can be inspected,
# documented, and tuned in one place without touching the scoring logic.

# Resting HR is considered elevated when it exceeds the personal 7-day mean
# by more than this many bpm. 5 bpm is a commonly used sports-science threshold.
RESTING_HR_ELEVATION_BPM = 5.0

# HRV is considered suppressed when it falls below the personal 7-day mean
# by more than this many milliseconds.
HRV_SUPPRESSION_MS = 5.0

# Training load is considered high when accumulated workout minutes over the
# past 3 days exceed this value (120 min = 2 hours).
HIGH_LOAD_3D_MINUTES = 120.0


def compute_risk_score(row: pd.Series) -> int:
    """
    Computes a rule-based overload risk score for a single day.

    Each condition contributes a fixed number of points. Missing values
    (NaN) are treated conservatively: the corresponding rule is skipped
    rather than contributing to the score.

    Scoring rules:
        sleep_hours < 6                                   → +2
        fatigue >= 8                                      → +2
        soreness >= 8                                     → +2
        resting_hr_diff_from_7d_mean > threshold          → +1
        hrv_diff_from_7d_mean < -threshold                → +1
        workout_load_3d_sum > HIGH_LOAD_3D_MINUTES        → +2

    Returns an integer score in the range [0, 10].
    """
    score = 0

    if pd.notna(row.get("sleep_hours")) and row["sleep_hours"] < 6:
        score += 2

    if pd.notna(row.get("fatigue")) and row["fatigue"] >= 8:
        score += 2

    if pd.notna(row.get("soreness")) and row["soreness"] >= 8:
        score += 2

    if (
        pd.notna(row.get("resting_hr_diff_from_7d_mean"))
        and row["resting_hr_diff_from_7d_mean"] > RESTING_HR_ELEVATION_BPM
    ):
        score += 1

    if (
        pd.notna(row.get("hrv_diff_from_7d_mean"))
        and row["hrv_diff_from_7d_mean"] < -HRV_SUPPRESSION_MS
    ):
        score += 1

    if (
        pd.notna(row.get("workout_load_3d_sum"))
        and row["workout_load_3d_sum"] > HIGH_LOAD_3D_MINUTES
    ):
        score += 2

    return score


def assign_risk_level(score: int) -> str:
    """
    Maps a numeric risk score to a categorical risk level.

        0–2  → "low"
        3–5  → "medium"
        6+   → "high"
    """
    if score <= 2:
        return "low"
    if score <= 5:
        return "medium"
    return "high"


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies compute_risk_score and assign_risk_level to every row in a DataFrame.

    Adds two columns:
        risk_score – integer score per day
        risk_level – categorical label ("low", "medium", "high")

    The DataFrame must already contain the feature columns produced by
    feature_engineering.add_all_features().
    """
    df = df.copy()
    df["risk_score"] = df.apply(compute_risk_score, axis=1)
    df["risk_level"] = df["risk_score"].apply(assign_risk_level)
    return df
