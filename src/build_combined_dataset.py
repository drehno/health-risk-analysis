import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from config import DATA_PROCESSED
from feature_engineering import add_all_features


def load_daily_metrics() -> pd.DataFrame:
    """Loads the processed Apple Health daily metrics."""
    path = DATA_PROCESSED / "daily_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_daily_metrics.py first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    return df


def load_manual_inputs() -> pd.DataFrame:
    """
    Loads manually recorded subjective wellness data.

    Expected columns: date, fatigue, soreness, readiness, bjj, lifting, intensity
    Scale: 1–10 for fatigue/soreness/readiness/intensity; 0/1 for bjj/lifting.
    """
    path = DATA_PROCESSED / "manual_inputs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create and populate the file first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date

    expected = {"date", "fatigue", "soreness", "readiness", "bjj", "lifting", "intensity"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in manual_inputs.csv: {missing}")

    return df


def merge_datasets(df_health: pd.DataFrame, df_manual: pd.DataFrame) -> pd.DataFrame:
    """
    Merges Apple Health metrics with manual inputs on date (inner join).

    Inner join is intentional: only days where both sources are present are
    included, avoiding rows with structurally missing subjective labels.
    """
    df = pd.merge(df_health, df_manual, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    print(
        f"{len(df_health)} health days + {len(df_manual)} manual days "
        f"→ {len(df)} shared days after merge."
    )
    return df


def save_combined(df: pd.DataFrame) -> None:
    """Saves the combined dataset with baselines to CSV."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / "combined_dataset.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    df_health = load_daily_metrics()
    df_manual = load_manual_inputs()

    df = merge_datasets(df_health, df_manual)
    df = add_all_features(df)

    print("\nFirst rows with baselines:")
    print(df.head(10).to_string())

    print("\nMissing values after baseline computation:")
    print(df.isna().sum())

    save_combined(df)
