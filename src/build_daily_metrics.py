import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import DATA_PROCESSED
from parse_health_xml import extract_records, filter_records

RESTING_HR = "HKQuantityTypeIdentifierRestingHeartRate"
HRV        = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
SLEEP      = "HKCategoryTypeIdentifierSleepAnalysis"
EXERCISE   = "HKQuantityTypeIdentifierAppleExerciseTime"

SLEEP_ASLEEP_VALUES = {
    "HKCategoryValueSleepAnalysisAsleep",
    "HKCategoryValueSleepAnalysisAsleepCore",
    "HKCategoryValueSleepAnalysisAsleepDeep",
    "HKCategoryValueSleepAnalysisAsleepREM",
}


def records_to_df(records: list) -> pd.DataFrame:
    """Converts a list of record dicts to a DataFrame with typed columns."""
    if not records:
        return pd.DataFrame(columns=["type", "value", "unit", "startDate", "endDate", "date"])

    df = pd.DataFrame(records)
    df["startDate"] = pd.to_datetime(df["startDate"], utc=False)
    df["date"] = df["startDate"].dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def build_sleep_series(records: list) -> pd.Series:
    """
    Computes daily sleep duration in hours from sleep analysis records.

    Apple Health sleep records use a category string as the value
    (e.g. HKCategoryValueSleepAnalysisAsleep), not a numeric duration.
    Sleep hours are therefore derived from endDate - startDate.
    Only asleep stages are counted; InBed and Awake are excluded.
    The date is taken from endDate (= morning of wake-up).
    """
    sleep_recs = filter_records(records, SLEEP)
    rows = []

    for r in sleep_recs:
        if r.get("value") not in SLEEP_ASLEEP_VALUES:
            continue
        try:
            start = pd.to_datetime(r["startDate"])
            end   = pd.to_datetime(r["endDate"])
        except Exception:
            continue

        duration_h = (end - start).total_seconds() / 3600
        if duration_h <= 0:
            continue

        rows.append({"date": end.date(), "sleep_hours": duration_h})

    if not rows:
        return pd.Series(dtype=float, name="sleep_hours")

    df = pd.DataFrame(rows)
    return df.groupby("date")["sleep_hours"].sum()


def build_daily_metrics(records: list) -> pd.DataFrame:
    """
    Aggregates raw Apple Health records into one row per day.

    Columns: date, resting_hr, hrv, sleep_hours, workout_minutes
    """
    df_hr = records_to_df(filter_records(records, RESTING_HR))
    daily_hr = (
        df_hr.groupby("date")["value"].mean().rename("resting_hr")
        if not df_hr.empty else pd.Series(dtype=float, name="resting_hr")
    )

    df_hrv = records_to_df(filter_records(records, HRV))
    daily_hrv = (
        df_hrv.groupby("date")["value"].mean().rename("hrv")
        if not df_hrv.empty else pd.Series(dtype=float, name="hrv")
    )

    daily_sleep = build_sleep_series(records)

    df_ex = records_to_df(filter_records(records, EXERCISE))
    daily_ex = (
        df_ex.groupby("date")["value"].sum().rename("workout_minutes")
        if not df_ex.empty else pd.Series(dtype=float, name="workout_minutes")
    )

    df_daily = pd.concat(
        [daily_hr, daily_hrv, daily_sleep, daily_ex], axis=1
    ).reset_index()
    df_daily = df_daily.rename(columns={"index": "date"})
    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    print(f"Built daily metrics: {len(df_daily)} days, {df_daily.shape[1]} columns.")
    print(f"Missing values:\n{df_daily.isna().sum()}")

    return df_daily


def save_daily_metrics(df: pd.DataFrame) -> None:
    """Saves the daily metrics table to CSV."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "daily_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    records = extract_records()
    df_daily = build_daily_metrics(records)
    print(df_daily.head(10))
    save_daily_metrics(df_daily)
