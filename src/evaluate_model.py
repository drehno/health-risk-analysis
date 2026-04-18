import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot to work without a display
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

RISK_LEVELS = ["low", "medium", "high"]


def print_report(y_true, y_pred) -> None:
    """
    Prints a per-class classification report.

    Precision and recall for the "high" class are the primary metrics here:
    missing a high-risk day (false negative) is more costly than a false alarm.
    """
    print(classification_report(y_true, y_pred, labels=RISK_LEVELS, zero_division=0))


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    """
    Renders and saves a confusion matrix to disk.

    Labels are ordered low → medium → high so the matrix reads from
    least to most severe along both axes.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=RISK_LEVELS,
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix – Risk Level Prediction")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_path}")
