# California Housing Price Prediction — Datheon Internship Task 2

## Project Overview

An end-to-end supervised machine learning regression project predicting median house values using the California Housing dataset.

This project compares three regression algorithms:

- Ordinary Least Squares (OLS)
- Ridge Regression (L2)
- Lasso Regression (L1)

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-Learn
- GridSearchCV
- Jupyter Notebook

## Machine Learning Workflow

1. Dataset loading and exploration
2. Missing-value and duplicate analysis
3. Correlation analysis
4. Feature engineering
5. Median imputation
6. Numerical feature scaling
7. Categorical feature encoding
8. OLS baseline model
9. Ridge and Lasso regression
10. Five-fold cross-validation and hyperparameter tuning
11. RMSE, MAE and R² evaluation
12. Predicted vs actual visualization
13. Residual error analysis

## Dataset

California Housing dataset containing 20,640 observations and 10 columns.

Target variable: `median_house_value`.

## Installation

```bash
pip install -r requirements.txt
```

## Execution

```bash
jupyter notebook
```

Open `Datheon_Task2_California_Housing_Regression.ipynb` and run all cells.

## Evaluation

The models are compared using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)


## Author

Azmain Haq Siam

Datheon Machine Learning & Data Science Internship

Application ID: DT-2026-YN1B7