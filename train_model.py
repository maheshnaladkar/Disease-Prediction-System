from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "disease_symptoms.csv"
TEST_PATH = ROOT / "dataset" / "testing.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    if "Disease" not in data.columns:
        raise ValueError("Training dataset must include a 'Disease' target column.")
    data = data.dropna(subset=["Disease"])
    data["Disease"] = data["Disease"].astype(str).str.strip()
    symptoms = [c for c in data.columns if c != "Disease"]
    if not symptoms or data.empty:
        raise ValueError("Dataset is empty or has no symptom features.")

    X = data[symptoms].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any():
        raise ValueError("Training data contains missing symptom values.")

    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X, data["Disease"])

    metrics = {
        "dataset_rows": int(len(data)),
        "feature_count": int(len(symptoms)),
        "class_count": int(data["Disease"].nunique()),
        "training_classes": sorted(data["Disease"].unique().tolist()),
        "note": "Educational symptom dataset only; model output is not a medical diagnosis."
    }

    if TEST_PATH.exists():
        test = pd.read_csv(TEST_PATH)
        if "Disease" in test.columns:
            test = test.dropna(subset=["Disease"])
            test["Disease"] = test["Disease"].astype(str).str.strip()
            # Align test data to the exact feature order used for training.
            for col in symptoms:
                if col not in test.columns:
                    test[col] = 0
            Xt = test[symptoms].apply(pd.to_numeric, errors="coerce").fillna(0)
            y_test = test["Disease"]
            predictions = model.predict(Xt)
            labels = sorted(set(y_test.tolist()) | set(predictions.tolist()))
            metrics.update({
                "test_rows": int(len(test)),
                "accuracy": float(accuracy_score(y_test, predictions)),
                "classification_report": classification_report(
                    y_test, predictions, labels=labels, output_dict=True, zero_division=0
                ),
                "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
                "labels": labels,
                "evaluation_note": "Score is based on the bundled Testing.csv sample and is not evidence of clinical accuracy."
            })

    joblib.dump({"model": model, "symptoms": symptoms}, MODEL_DIR / "disease_model.pkl")
    (MODEL_DIR / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Training complete.")
    print(f"Training rows: {len(data)} | Symptoms: {len(symptoms)} | Disease labels: {data['Disease'].nunique()}")
    if "accuracy" in metrics:
        print(f"Testing.csv accuracy: {metrics['accuracy']:.3f} (educational dataset only)")
    print(f"Saved model to {MODEL_DIR / 'disease_model.pkl'}")
    print("This model is for education only, not diagnosis or treatment.")

if __name__ == "__main__":
    main()
