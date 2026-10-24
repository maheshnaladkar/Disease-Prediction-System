from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "disease_symptoms.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if "Disease" not in data.columns:
        raise ValueError("Dataset must include a 'Disease' target column.")
    if data.empty or data["Disease"].isna().any():
        raise ValueError("Dataset is empty or contains missing target labels.")

    X = data.drop(columns=["Disease"])
    y = data["Disease"].astype(str)
    if X.empty:
        raise ValueError("Dataset has no symptom feature columns.")
    if X.isna().any().any():
        raise ValueError("Please handle missing symptom values before training.")
    X = X.apply(pd.to_numeric, errors="raise")

    # For a real study, choose a dataset large enough for every class and use a
    # carefully designed validation strategy. This bundled dataset is synthetic.
    counts = y.value_counts()
    stratify = y if len(counts) > 1 and counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(y.unique().tolist())
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump({"model": model, "symptoms": list(X.columns)}, MODEL_DIR / "disease_model.pkl")
    metrics = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
        "dataset_rows": int(len(data)),
        "feature_count": int(X.shape[1]),
        "note": "Synthetic demonstration data only; not clinically meaningful."
    }
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Training complete. Demo split accuracy: {accuracy:.3f}")
    print(f"Saved model to {MODEL_DIR / 'disease_model.pkl'}")
    print("Reminder: synthetic demo results are not evidence of medical accuracy.")

if __name__ == "__main__":
    main()
