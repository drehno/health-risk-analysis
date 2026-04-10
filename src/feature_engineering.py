import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def add_resting_hr_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 7-day rolling mean and deviation for resting heart rate.

    Columns added:
        resting_hr_7d_mean       – rolling 7-day mean (min 3 observations)
        resting_hr_diff_from_7d_mean – deviation from that mean

    Using min_periods=3 so the baseline only activates once enough
    history is available to make it meaningful.
    """
    df["resting_hr_7d_mean"] = df["resting_hr"].rolling(window=7, min_periods=3).mean()
    df["resting_hr_diff_from_7d_mean"] = df["resting_hr"] - df["resting_hr_7d_mean"]
    return df


def add_hrv_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 7-day rolling mean and deviation for HRV.

    Columns added:
        hrv_7d_mean              – rolling 7-day mean (min 3 observations)
        hrv_diff_from_7d_mean    – deviation from that mean

    HRV deviation is particularly informative: a drop below personal baseline
    is a stronger recovery signal than any absolute HRV value.
    """
    df["hrv_7d_mean"] = df["hrv"].rolling(window=7, min_periods=3).mean()
    df["hrv_diff_from_7d_mean"] = df["hrv"] - df["hrv_7d_mean"]
    return df


def add_sleep_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Adds a rolling mean of sleep hours.

    Columns added:
        sleep_3d_avg – rolling mean over `window` days (default 3)

    min_periods=1 so the feature is available from day 1.
    """
    df[f"sleep_{window}d_avg"] = df["sleep_hours"].rolling(window=window, min_periods=1).mean()
    return df


def add_training_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cumulative training load over 3-day and 7-day windows.

    Columns added:
        workout_load_3d_sum – total workout minutes over the past 3 days
        workout_load_7d_sum – total workout minutes over the past 7 days

    Both use min_periods=1 so no rows are lost at the start of the dataset.
    The 3-day sum captures acute load; the 7-day sum captures chronic load.
    """
    df["workout_load_3d_sum"] = df["workout_minutes"].rolling(window=3, min_periods=1).sum()
    df["workout_load_7d_sum"] = df["workout_minutes"].rolling(window=7, min_periods=1).sum()
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps in sequence.

    Expects the DataFrame to be sorted by date (ascending) before calling.
    Returns a new DataFrame; the input is not modified.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    df = add_resting_hr_features(df)
    df = add_hrv_features(df)
    df = add_sleep_features(df)
    df = add_training_load_features(df)

    return df
