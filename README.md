# Datheon Internship Task 1 — Iris Classification Benchmark

**Applicant:** Azmain Haq Siam  
**Application ID:** DT-2026-YN1B7

## Project overview
This project implements a reproducible multi-class classification workflow on the Iris dataset. It compares Logistic Regression and Decision Tree models using an 80/20 stratified train/test split, StandardScaler pipelines, classification metrics, confusion matrices, exploratory analysis, and two-feature decision-boundary plots.

## Repository structure
```text
datheon_task1_iris_classification/
├── Datheon_Task1_Iris_Classification.ipynb
├── data/
│   └── iris.csv
├── outputs/                 
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements
- Python 3.10 or newer recommended
- Jupyter Notebook or JupyterLab
- Packages listed in `requirements.txt`

## Run locally
### 1. Extract the project
Unzip the project folder if you downloaded the ZIP archive.

### 2. Open a terminal in the project directory
```bash
cd datheon_task1_iris_classification
```

### 3. Create and activate a virtual environment :

Windows Command Prompt:
```bat
py -m venv .venv
.venv\Scripts\activate.bat
```

### 4. Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Launch Jupyter
```bash
jupyter notebook
```

## Methodology
1. Load the bundled Iris CSV with Pandas.
2. Inspect shape, dtypes, missing values, duplicate rows, descriptive statistics, and class distribution.
3. Visualize feature relationships with a Seaborn pairplot.
4. Split the data into 80% training and 20% testing using stratification and a fixed random seed.
5. Fit StandardScaler only on training data through a scikit-learn Pipeline.
6. Train Logistic Regression and Decision Tree models.
7. Compare accuracy, macro precision, macro recall, and macro F1-score on held-out test data.
8. Display classification reports, confusion-matrix heatmaps, metric comparison, and two-feature decision boundaries.
9. Export the benchmark table to `outputs/benchmark_metrics.csv`.
