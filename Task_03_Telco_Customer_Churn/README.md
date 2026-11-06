# Datheon Internship — Task 3: Predictive Customer Churn Classifier

## Objective
Predict telecom customer churn and compare Random Forest and XGBoost. The assignment target is ROC-AUC > 0.80; this is a goal, not a guaranteed result. Report only the measured hold-out score.

## Dataset
IBM Telco Customer Churn CSV:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

The notebook attempts automatic download. If unavailable, download the CSV manually and save as `data/Telco-Customer-Churn.csv`. **The CSV is not included in this ZIP.**

## Structure
```text
Datheon_Task3_Telco_Customer_Churn/
├── Datheon_Task3_Telco_Customer_Churn.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/DATASET_INSTRUCTIONS.txt
└── outputs/.gitkeep
```

## Windows setup
```bat
py -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```
Open the notebook and use **Kernel → Restart & Run All**.

## Methodology
- Clean `TotalCharges`; convert invalid/blank values to missing numeric values.
- Map churn No=0 and Yes=1; exclude customerID.
- Stratified 80/20 train/test split.
- Numeric median imputation/scaling and categorical imputation/one-hot encoding.
- SMOTE inside imbalanced-learn pipelines to avoid cross-validation leakage.
- Five-fold StratifiedKFold and RandomizedSearchCV for both models.
- Test ROC-AUC, accuracy, precision, recall and F1.
- ROC curves, confusion matrices, classification reports and top-10 feature importance.
- Optional threshold selected from training-only out-of-fold predictions.

## Outputs created after notebook execution
`outputs/model_metrics.csv`, `outputs/test_predictions.csv`, `outputs/feature_importance.csv`, `outputs/training_oof_threshold_analysis.csv`.

## Notes
Fixed random seed 42 is used. Runtime and results depend on environment and package versions. Feature importance is not causal. No performance score is claimed before execution.
