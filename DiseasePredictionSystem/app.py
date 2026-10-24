from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "disease_model.pkl"
DATA_PATH = ROOT / "dataset" / "disease_symptoms.csv"

st.set_page_config(page_title="Disease Prediction System", page_icon="🩺", layout="wide")

st.title("🩺 Disease Prediction System")
st.caption("Educational machine-learning demonstration — not a medical diagnostic tool.")

with st.sidebar:
    st.header("About")
    st.write("Select symptom indicators from the demo dataset to see the output of a trained classification model.")
    st.warning("Do not use this application to diagnose, treat, or rule out any health condition.")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Open the project folder in a terminal and run: `python train_model.py`")
    st.stop()

try:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    symptoms = bundle["symptoms"]
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

tab_predict, tab_about, tab_metrics = st.tabs(["Symptom Demo", "Project Information", "Model Evaluation"])

with tab_predict:
    st.subheader("Select the available symptom indicators")
    st.write("The dataset is illustrative and intentionally small. These predictions are not clinically validated.")
    search = st.text_input("Search symptoms", placeholder="Type to filter symptom names")
    visible = [s for s in symptoms if search.lower() in s.replace("_", " ").lower()]
    selected = []
    cols = st.columns(3)
    for i, symptom in enumerate(visible):
        label = symptom.replace("_", " ").title()
        with cols[i % 3]:
            if st.checkbox(label, key=f"symptom_{symptom}"):
                selected.append(symptom)

    left, right = st.columns([1, 2])
    with left:
        if st.button("Clear selections"):
            for symptom in symptoms:
                st.session_state.pop(f"symptom_{symptom}", None)
            st.rerun()
    with right:
        st.caption(f"{len(selected)} symptom indicator(s) selected")

    if st.button("Run demo prediction", type="primary", use_container_width=True):
        if not selected:
            st.warning("Select at least one symptom indicator to run the demonstration.")
        else:
            row = pd.DataFrame([[int(s in selected) for s in symptoms]], columns=symptoms)
            prediction = model.predict(row)[0]
            st.subheader("Model output")
            st.info(f"Predicted dataset category: **{prediction}**")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(row)[0]
                result = pd.DataFrame({
                    "Dataset category": model.classes_,
                    "Model score": probabilities
                }).sort_values("Model score", ascending=False)
                st.write("Model scores (not medical probabilities or diagnostic confidence)")
                st.bar_chart(result.set_index("Dataset category"))
            st.warning(
                "This is an educational output from an illustrative dataset. It cannot confirm or rule out "
                "a disease. Do not make health decisions from this result; consult a qualified healthcare professional."
            )

with tab_about:
    st.subheader("Project overview")
    st.write(
        "This project demonstrates a basic supervised classification workflow: a labelled CSV dataset is "
        "loaded, a Random Forest classifier is trained, and a Streamlit interface sends selected feature values "
        "to the saved model."
    )
    st.markdown("""
    **Technology stack**
    - Python
    - pandas and NumPy for data handling
    - scikit-learn for model training and evaluation
    - joblib for saving/loading the model
    - Streamlit for the web interface

    **Important limitation:** the bundled data is synthetic, created only to demonstrate the software workflow. It is not medical evidence and the model is not suitable for clinical use.
    """)

with tab_metrics:
    metrics_path = ROOT / "models" / "evaluation.json"
    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.subheader("Held-out demo evaluation")
        st.metric("Accuracy on demo test split", f"{metrics.get('accuracy', 0):.1%}")
        st.write("Classification report")
        st.json(metrics.get("classification_report", {}))
        st.write("Confusion matrix")
        st.dataframe(pd.DataFrame(metrics.get("confusion_matrix", []),
                                  index=metrics.get("labels", []),
                                  columns=metrics.get("labels", [])))
        st.caption("Metrics are for synthetic demonstration data only and do not indicate real-world medical performance.")
    else:
        st.info("Evaluation results not found. Run `python train_model.py` to generate them.")
