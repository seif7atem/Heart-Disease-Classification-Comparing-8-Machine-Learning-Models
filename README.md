# ❤️ Heart Disease Classification

A machine learning project that predicts the presence of heart disease from patient clinical features, comparing 8 classifiers end to end.

> ⚠️ For educational purposes only. Not a substitute for professional medical diagnosis.

## Pipeline
1. **EDA:** duplicate removal, summary stats, target distribution, disease rate by age/sex/chest pain type, correlation matrix
2. **Outlier treatment:** Z-score (|z| > 3) on `trestbps`, `chol`, `thalach`, `oldpeak`, replaced with the median
3. **Feature engineering:** age groups (one-hot) and binary sex encoding
4. **Model comparison:** stratified 10-fold CV on scaled (MinMax) and raw data
5. **Final evaluation:** top 3 models on a holdout set (Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices)
6. **Prediction demo** with the best model

## Models Compared
Logistic Regression · KNN · Decision Tree · Random Forest · Extra Trees · SVM (RBF) · Gradient Boosting · XGBoost

## Installation
```bash
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn
```

## Usage
Place `heart.csv` next to the script, then run:
```bash
python heart_classification.py
```
All plots and `final_evaluation.csv` are saved to the `output/` folder.

## Dataset
Heart Disease dataset (Cleveland/UCI), 1025 records, 13 features + binary target. Not included in this repo.

## Results
| Model | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| XX | XX | XX | XX | XX |

Best model: **XX**

## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · SciPy · Matplotlib · Seaborn
