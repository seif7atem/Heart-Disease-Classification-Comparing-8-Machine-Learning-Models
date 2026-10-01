# -*- coding: utf-8 -*-
"""
Heart Disease Classification
============================
A single self-contained script that performs:
  1. Data exploration (EDA)
  2. Outlier treatment
  3. Feature engineering (age groups, one-hot encoding)
  4. Scaling + encoding
  5. Training and comparing 8 classifiers via 10-fold CV
  6. Final holdout evaluation + prediction demo

Dataset   : heart.csv (Cleveland/Harvard heart dataset, 1025 records)
"""
import os
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ============================================================
# Global settings
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
RANDOM_STATE = 42


def load_data():
    """Read the CSV file (first row is the header)."""
    df = pd.read_csv(DATA_PATH, header=0)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def explore(df):
    print("=" * 70)
    print("Step 1: Exploratory Data Analysis (EDA)")
    print("=" * 70)
    print(f"Dataset shape (rows, cols): {df.shape}")
    n_dup = df.duplicated().sum()
    print(f"Exact duplicate rows: {n_dup}")

    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Rows after dropping duplicates: {len(df)}")

    print("\nGeneral statistics:")
    print(df.describe().T.to_string())

    print("\nColumn data types:")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    print("\nTarget distribution:")
    print(df["target"].value_counts().to_string())
    print(f"Rate of patients with heart disease: {df['target'].mean() * 100:.2f}%")
    return df


def plot_eda(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Exploratory Data Analysis (EDA)", fontsize=14)

    labels = ["No disease (0)", "Disease (1)"]
    axes[0, 0].bar(labels, df["target"].value_counts().values,
                   color=["#e74c3c", "#2ecc71"])
    axes[0, 0].set_title("Target distribution")

    bins = np.histogram_bin_edges(df["age"], bins=15)
    axes[0, 1].hist(df.loc[df["target"] == 0, "age"], bins=bins,
                    alpha=0.7, label="No disease", color="#e74c3c")
    axes[0, 1].hist(df.loc[df["target"] == 1, "age"], bins=bins,
                    alpha=0.7, label="Disease", color="#2ecc71")
    axes[0, 1].set_title("Age by target")
    axes[0, 1].legend()

    sex_rate = df.groupby("sex")["target"].mean() * 100
    axes[1, 0].bar(["Female (0)", "Male (1)"], sex_rate.values,
                   color=["#3498db", "#f39c12"])
    axes[1, 0].set_title("Disease rate by sex (%)")
    axes[1, 0].set_ylim(0, 100)

    cp_rate = df.groupby("cp")["target"].mean() * 100
    axes[1, 1].bar(["Typical (0)", "Atypical (1)", "Non-anginal (2)",
                    "Asymptomatic (3)"], cp_rate.values, color="#9b59b6")
    axes[1, 1].set_title("Disease rate by chest pain type (%)")
    axes[1, 1].set_ylim(0, 100)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_plots.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"EDA plots saved to: {path}")
    return path


def plot_correlation(df):
    corr_matrix = df.corr()

    plt.figure(figsize=(13, 11))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, square=True)
    plt.title("Correlation matrix")
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    plt.savefig(path1, dpi=120)
    plt.close()
    print(f"Correlation matrix saved to: {path1}")

    corr_target = corr_matrix["target"].drop("target").sort_values(ascending=False)
    plt.figure(figsize=(9, 9))
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in corr_target.values]
    corr_target.plot(kind="barh", color=colors)
    plt.title("Feature correlation with target")
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "feature_correlation.png")
    plt.savefig(path2, dpi=120)
    plt.close()
    print(f"Feature correlation saved to: {path2}")
    return path2


def treat_outliers(df):
    print("\n" + "=" * 70)
    print("Step 2: Treating outliers (Z-Score > 3 -> median)")
    print("=" * 70)
    df = df.copy()
    numeric_cols = ["trestbps", "chol", "thalach", "oldpeak"]
    total = 0
    for col in numeric_cols:
        z = np.abs(stats.zscore(df[col]))
        outliers = z > 3
        count = int(outliers.sum())
        total += count
        if count > 0:
            median = df.loc[~outliers, col].median()
            df.loc[outliers, col] = median
            print(f"  Column {col}: replaced {count} outlier(s) with median")
    print(f"Total outliers treated: {total}")
    return df


def encode_and_prepare(df):
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 40, 50, 60, 70, 200],
        labels=["<40", "40-49", "50-59", "60-69", ">=70"],
    )
    df = pd.get_dummies(df, columns=["age_group"], drop_first=True)
    df["sex_bin"] = (df["sex"] == 1).astype(int)
    df = df.drop(columns=["sex"], errors="ignore")
    return df


def build_models():
    return OrderedDict([
        ("Logistic Regression",
         LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("KNN", KNeighborsClassifier(n_neighbors=7)),
        ("Decision Tree",
         DecisionTreeClassifier(random_state=RANDOM_STATE,
                                min_samples_leaf=5)),
        ("Random Forest",
         RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ("Extra Trees",
         ExtraTreesClassifier(n_estimators=300, random_state=RANDOM_STATE)),
        ("SVM (RBF)", SVC(probability=True, random_state=RANDOM_STATE)),
        ("Gradient Boosting",
         GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ("XGBoost", XGBClassifier(eval_metric="logloss",
                                  random_state=RANDOM_STATE)),
    ])


def evaluate_cv(models, X, y):
    print("\n" + "=" * 70)
    print("Model comparison using Cross-Validation (10 folds, accuracy)")
    print("=" * 70)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    results = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        results.append((scores.mean(), scores.std(), name))

    results.sort(reverse=True)
    for i, (mean, std, name) in enumerate(results, 1):
        print(f"  {i}. {name:<22} | Mean: {mean:.4f} | Std: {std:.4f}")
    return results


def final_evaluation(summary, ordered_models, X_train, y_train, X_test, y_test):
    print("\n" + "=" * 70)
    print("Final evaluation on holdout test set")
    print("=" * 70)
    for name, model in ordered_models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        summary.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4),
        })

        print(f"\n### {name}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")
        print(f"ROC-AUC  : {roc:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["No disease",
                                                  "Disease"]))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No disease", "Disease"],
                    yticklabels=["No disease", "Disease"])
        plt.title(f"Confusion matrix - {name}")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR,
                            f"confusion_matrix_{name.replace(' ', '_')}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"Confusion matrix saved to: {path}")

    summary_df = pd.DataFrame(summary).sort_values("Accuracy",
                                                   ascending=False)
    return summary_df


def quick_prediction_demo(best_model, scaler, X_full_scaled, y_full,
                          feature_cols):
    print("\n" + "=" * 70)
    print("Quick prediction demo (best model, trained on full data)")
    print("=" * 70)
    best_model.fit(X_full_scaled, y_full)

    sample = np.median(X_full_scaled, axis=0)
    sample_df = pd.DataFrame([sample], columns=feature_cols)
    proba = best_model.predict_proba(sample_df)[0][1]

    print("Demo sample (median of all features):")
    print(sample_df.T.to_string())
    print()
    print(f"Disease probability: {proba * 100:.2f}%")
    print(f"Final decision: {'Disease' if proba >= 0.5 else 'No disease'}")


def main():
    print("=" * 70)
    print("Heart Disease Classification Project")
    print("=" * 70)

    df = load_data()
    df = explore(df)

    plot_eda(df)
    plot_correlation(df)

    df = treat_outliers(df)
    df = encode_and_prepare(df)

    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()

    print("\n" + "=" * 70)
    print("Phase 1: Scaled data (MinMaxScaler)")
    print("=" * 70)
    cv_scaled = evaluate_cv(models, X_train_scaled, y_train)
    best_scaled = [name for _, _, name in cv_scaled[:3]]

    print("\n" + "=" * 70)
    print("Phase 2: Raw data (no scaling)")
    print("=" * 70)
    cv_raw = evaluate_cv(models, X_train, y_train)
    best_raw = [name for _, _, name in cv_raw[:3]]

    print("\n" + "=" * 70)
    print("Choosing best models for final evaluation")
    print("=" * 70)
    print(f"Top-3 from scaled data: {best_scaled}")
    print(f"Top-3 from raw data:    {best_raw}")

    chosen = best_scaled
    chosen_models = OrderedDict((n, models[n]) for n in chosen)
    summary_final = final_evaluation([], list(chosen_models.items()),
                                     X_train_scaled, y_train,
                                     X_test_scaled, y_test)

    print("\nFinal evaluation summary (sorted by accuracy):")
    print(summary_final.to_string(index=False))
    path_summary = os.path.join(OUTPUT_DIR, "final_evaluation.csv")
    summary_final.to_csv(path_summary, index=False)
    print(f"Summary saved to: {path_summary}")

    quick_prediction_demo(chosen_models[chosen[0]], scaler,
                          X_train_scaled, y_train, feature_cols)

    print("\n" + "=" * 70)
    print("Project finished. All outputs are in the output/ folder")
    print("=" * 70)


if __name__ == "__main__":
    main()
