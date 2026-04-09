import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from config import DATA_PROCESSED
from src.parse_health_xml import extract_records, filter_records

# Relevante HK-Typen
RESTING_HR  = "HKQuantityTypeIdentifierRestingHeartRate"
HRV         = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
SLEEP       = "HKCategoryTypeIdentifierSleepAnalysis"
EXERCISE    = "HKQuantityTypeIdentifierAppleExerciseTime"


def records_to_df(records):
    """Wandelt eine Liste von Record-Dicts in einen DataFrame um."""
    df = pd.DataFrame(records)
    df["startDate"] = pd.to_datetime(df["startDate"], utc=False)
    df["date"] = df["startDate"].dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def build_daily_metrics(records):
    """
    Baut eine tägliche Zusammenfassung aus den rohen Records.
    Gibt einen DataFrame mit einer Zeile pro Tag zurück.
    """

    # Ruhepuls – Tagesmittelwert
    df_hr = records_to_df(filter_records(records, RESTING_HR))
    daily_hr = df_hr.groupby("date")["value"].mean().rename("resting_hr")

    # HRV – Tagesmittelwert
    df_hrv = records_to_df(filter_records(records, HRV))
    daily_hrv = df_hrv.groupby("date")["value"].mean().rename("hrv")

    # Schlaf – Summe pro Tag in Stunden
    df_sleep = records_to_df(filter_records(records, SLEEP))
    daily_sleep = (
        df_sleep.groupby("date")["value"].sum() / 60
    ).rename("sleep_hours")

    # Workout-Minuten – Summe pro Tag
    df_ex = records_to_df(filter_records(records, EXERCISE))
    daily_ex = df_ex.groupby("date")["value"].sum().rename("workout_minutes")

    # Alles zusammenführen
    df_daily = pd.concat(
        [daily_hr, daily_hrv, daily_sleep, daily_ex], axis=1
    ).reset_index()

    df_daily = df_daily.sort_values("date").reset_index(drop=True)

    print(f"Tagestabelle gebaut: {len(df_daily)} Tage, {df_daily.shape[1]} Spalten.")
    print(f"Fehlende Werte:\n{df_daily.isna().sum()}")

    return df_daily


def save_daily_metrics(df):
    """Speichert die Tagestabelle als CSV."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    output_path = DATA_PROCESSED / "daily_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"Gespeichert: {output_path}")


if __name__ == "__main__":
    records = extract_records()
    df_daily = build_daily_metrics(records)
    print(df_daily.head(10))
    save_daily_metrics(df_daily)