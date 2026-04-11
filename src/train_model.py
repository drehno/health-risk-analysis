import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from config import DATA_PROCESSED
from feature_engineering import add_all_features
from risk_score import add_risk_score


FEATURES = [
    "sleep_hours",
    "hrv_diff_from_7d_mean",
    "resting_hr_diff_from_7d_mean",
    "workout_load_7d_sum",
    "fatigue",
    "soreness",
]
TARGET = "risk_level"

# Chronological split ratio: first 80% of days for training, last 20% for testing.
TRAIN_RATIO = 0.8

# Refuse to train on datasets smaller than this to avoid misleading results.
MIN_ROWS = 20


def load_dataset() -> pd.DataFrame:
    """
    Loads the combined dataset, applies feature engineering and risk scoring.

    Expects combined_dataset.csv to already contain merged Apple Health and
    manual input data. Run build_combined_dataset.py first if it does not exist.
    """
    path = DATA_PROCESSED / "combined_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_combined_dataset.py first."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df = df.sort_values("date").reset_index(drop=True)

    df = add_all_features(df)
    df = add_risk_score(df)

    return df


def split_chronological(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train and test sets in chronological order.

    A random split must not be used for time series data: it would allow the
    model to see future observations during training, making evaluation invalid.
    """
    split_idx = int(len(df) * TRAIN_RATIO)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_pipeline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> tuple[SimpleImputer, LogisticRegression, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Fits an imputer and a logistic regression classifier on the training set.

    The imputer is fit exclusively on training data and then applied to the
    test set, preventing any information from the test set from leaking into
    the training process.

    Returns imputer, model, X_train_imp, y_train, X_test_imp, y_test.
    """
    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]
    X_test  = df_test[FEATURES]
    y_test  = df_test[TARGET]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURES
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURES
    )

    # class_weight="balanced" compensates for the natural scarcity of high-risk
    # days in the dataset, preventing the model from simply predicting "low"
    # for everything and achieving high accuracy by doing so.
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train_imp, y_train)

    return imputer, model, X_train_imp, y_train, X_test_imp, y_test


def save_artifacts(imputer: SimpleImputer, model: LogisticRegression) -> None:
    """Saves the fitted imputer and model to disk for later use."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, DATA_PROCESSED / "imputer.joblib")
    joblib.dump(model,   DATA_PROCESSED / "model.joblib")
    print(f"Model and imputer saved to {DATA_PROCESSED}/")


def train(df: pd.DataFrame) -> tuple[SimpleImputer, LogisticRegression]:
    """
    Full training pipeline: split → impute → fit → save.

    Returns the fitted imputer and model.
    """
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Dataset has only {len(df)} rows. At least {MIN_ROWS} are required "
            "for a meaningful train/test split. Add more data to manual_inputs.csv."
        )

    df_train, df_test = split_chronological(df)
    print(f"Train: {len(df_train)} days | Test: {len(df_test)} days")
    print(f"Label distribution (train):\n{df_train[TARGET].value_counts()}")

    imputer, model, _, _, X_test_imp, y_test = build_pipeline(df_train, df_test)

    y_pred = model.predict(X_test_imp)
    print(f"\nTest accuracy: {(y_pred == y_test.values).mean():.2%}")
    print(f"Label distribution (test predictions): {pd.Series(y_pred).value_counts().to_dict()}")

    save_artifacts(imputer, model)
    return imputer, model


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset: {len(df)} days | Features: {FEATURES} | Target: {TARGET}")
    train(df)
