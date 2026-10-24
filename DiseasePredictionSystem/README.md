# Disease Prediction System Using Machine Learning

A beginner-friendly MCA project demonstrating a supervised classification workflow with Python, pandas, scikit-learn, joblib, and Streamlit.

## Important medical and dataset disclaimer

The included CSV uses **synthetic demonstration labels** (`Demo_Class_A`, etc.). These labels are not actual diseases, and the data is not clinical evidence. This project demonstrates software architecture and ML workflow only. It must not be used to diagnose, treat, or rule out a health condition. Do not describe the demo model as medically accurate.

## Features

- Symptom-feature checkbox interface
- Searchable feature list
- Random Forest classification model
- Model scores shown as model scores, not medical probabilities
- Held-out split evaluation and confusion matrix
- Project information and limitations
- Local Streamlit web interface

## Requirements

- Python 3.10 or later recommended
- Windows, macOS, or Linux
- Internet connection for first-time package installation

## Setup on Windows PowerShell

Open PowerShell in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

If PowerShell prevents activation, use the environment's Python directly:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The local app normally opens at http://localhost:8501.

## Project structure

```text
DiseasePredictionSystem/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── dataset/
│   └── disease_symptoms.csv
└── models/
    └── (generated after training)
```

## How it works

1. `train_model.py` loads and validates the CSV.
2. It separates symptom features from the `Disease` label.
3. It makes a train/test split and fits a Random Forest classifier.
4. It saves the model and evaluation metrics in `models/`.
5. `app.py` loads the saved model and encodes selected symptoms as 0/1 values.
6. The app displays the predicted synthetic category and evaluation information.

## Replace the demo dataset responsibly

If you replace the dataset, keep the `Disease` target column and make sure all feature columns are numeric 0/1 values, missing values are handled, and the target labels are valid. Use a documented dataset with appropriate permissions. For any real medical research, consult domain experts and use suitable validation and governance; this starter project is not a clinical tool.

## Troubleshooting

- **Model not found:** run `python train_model.py` before starting Streamlit.
- **Module not found:** activate `.venv` and run `pip install -r requirements.txt`.
- **Streamlit command not found:** run `python -m streamlit run app.py`.
- **Dataset error:** verify `dataset/disease_symptoms.csv` exists and includes the `Disease` column.
