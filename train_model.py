"""Train a Random Forest classifier for drowsiness detection using synthetic facial features."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "drowsiness_data.csv"
MODEL_PATH = PROJECT_DIR / "model.pkl"
OUTPUT_DIR = PROJECT_DIR / "outputs"
TARGET = "label"
FEATURES = [
    "left_ear",
    "right_ear",
    "avg_ear",
    "mouth_ratio",
    "head_pitch",
    "head_yaw",
    "eye_closure_frames",
    "blink_ratio",
]
FEATURE_LABELS = {
    "left_ear": "Left EAR",
    "right_ear": "Right EAR",
    "avg_ear": "Average EAR",
    "mouth_ratio": "Mouth Ratio",
    "head_pitch": "Head Pitch",
    "head_yaw": "Head Yaw",
    "eye_closure_frames": "Consecutive Closed-Eye Frames",
    "blink_ratio": "Blink Ratio",
}


def validate_dataset(data: pd.DataFrame) -> None:
    """Ensure the CSV includes the expected feature columns and target column."""
    missing_columns = [column for column in [*FEATURES, TARGET] if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data.empty:
        raise ValueError(f"Dataset at {DATA_PATH} is empty.")


def build_model() -> RandomForestClassifier:
    """Create the training model with a stable configuration for tabular ML data."""
    return RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )


def save_visualizations(importances: pd.Series, matrix: object) -> None:
    """Save feature importance and confusion matrix plots."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    ordered_importances = importances.sort_values(ascending=False)
    axis.bar(
        [FEATURE_LABELS[feature] for feature in ordered_importances.index],
        ordered_importances.values,
        color="#1f77b4",
    )
    axis.set_title("Random Forest Feature Importance")
    axis.set_xlabel("Feature")
    axis.set_ylabel("Importance")
    axis.grid(axis="y", linestyle=":", alpha=0.35)
    plt.xticks(rotation=25, ha="right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "feature_importance.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_xticks([0, 1], labels=["ALERT", "DROWSY"])
    axis.set_yticks([0, 1], labels=["ALERT", "DROWSY"])
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=11,
            )
    figure.colorbar(image, ax=axis, label="Number of samples")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Load data, train a Random Forest, and save the model."""
    data = pd.read_csv(DATA_PATH)
    validate_dataset(data)

    print("# =============================================")
    print("SOLDIER DROWSINESS DETECTION USING ML")
    print("Dataset:")
    print(f"Records: {len(data)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Class counts: {data[TARGET].value_counts().to_dict()}")

    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.20,
        random_state=42,
        stratify=data[TARGET],
    )

    print("\nModel: Random Forest Classifier")
    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")

    model = build_model()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    print("\n## --------------- MODEL PERFORMANCE ------------")
    print(f"Accuracy  : {accuracy:.2%}")
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print(f"F1 Score  : {f1:.2%}")
    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print("                 Predicted")
    print("             ALERT   DROWSY")
    print(f"Actual ALERT     {matrix[0, 0]:>4}      {matrix[0, 1]:>4}")
    print(f"Actual DROWSY    {matrix[1, 0]:>4}      {matrix[1, 1]:>4}")

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n--------------- FEATURE IMPORTANCE -----------")
    for rank, (feature, importance) in enumerate(importances.items(), start=1):
        print(f"{rank}. {FEATURE_LABELS[feature]} ({importance:.3f})")

    save_visualizations(importances, matrix)
    print("\nGraphs saved in outputs/:")
    print("- outputs/feature_importance.png")
    print("- outputs/confusion_matrix.png")

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved successfully as {MODEL_PATH.name}")
    print("This is an educational ML demo and not a medical or military safety system.")


if __name__ == "__main__":
    main()