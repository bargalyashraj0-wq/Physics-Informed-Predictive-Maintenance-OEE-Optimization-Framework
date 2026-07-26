"""
XGBoost Predictive Maintenance Pipeline + SHAP Interpretability
=================================================================
Two models trained on the physics-engineered feature set:
  1. XGBRegressor  -> Remaining Useful Life (RUL) in cycles
  2. XGBClassifier -> Tool Wear State (Healthy / Degraded / Critical)

Group-aware train/test split (by unit_id) prevents data leakage across a
machine's own life trajectory. Hyperparameters tuned via RandomizedSearchCV
(Bayesian-style random sampling over a defined search space — swap in
`skopt.BayesSearchCV` directly for true Bayesian optimization if the
`scikit-optimize` package is available in the target environment).
SHAP TreeExplainer quantifies each physics-derived feature's marginal
contribution to individual failure-risk predictions -> the evidence trail
an RCA (root cause analysis) engineer needs.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV, GroupKFold
from sklearn.metrics import (mean_absolute_error, r2_score, root_mean_squared_error,
                              accuracy_score, f1_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

DATA_PATH = "/home/claude/pdm_project/data/features_engineered.csv"
FIG_DIR = "/home/claude/pdm_project/outputs/figures"
REPORT_DIR = "/home/claude/pdm_project/outputs/reports"

DROP_COLS = ["unit_id", "cycle", "RUL", "tool_state"]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    return df, feature_cols


def group_split(df, feature_cols, target_col, test_size=0.25, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["unit_id"]))
    X_train, X_test = df.iloc[train_idx][feature_cols], df.iloc[test_idx][feature_cols]
    y_train, y_test = df.iloc[train_idx][target_col], df.iloc[test_idx][target_col]
    groups_train = df.iloc[train_idx]["unit_id"]
    return X_train, X_test, y_train, y_test, groups_train


# ---------------------------------------------------------------------------
# 1. RUL REGRESSION
# ---------------------------------------------------------------------------
def train_rul_regressor(df, feature_cols):
    X_train, X_test, y_train, y_test, groups_train = group_split(df, feature_cols, "RUL")

    param_dist = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    base = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    cv = GroupKFold(n_splits=4)
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=25, cv=cv,
        scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train, groups=groups_train)
    best_model = search.best_estimator_

    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics = {"MAE_cycles": round(mae, 2), "RMSE_cycles": round(rmse, 2),
               "R2": round(r2, 4), "best_params": search.best_params_}
    print("\n=== RUL Regressor (XGBoost) ===")
    print(json.dumps(metrics, indent=2))

    # --- Plot: predicted vs actual RUL ---
    plt.figure(figsize=(5.5, 5))
    plt.scatter(y_test, preds, alpha=0.35, s=14, color="#2563eb")
    lims = [0, max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", linewidth=1.2, label="Ideal (y = x)")
    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title(f"RUL Prediction: XGBoost Regressor\nMAE={mae:.1f} cycles | R²={r2:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/rul_pred_vs_actual.png")
    plt.close()

    return best_model, X_train, X_test, y_test, preds, metrics


# ---------------------------------------------------------------------------
# 2. TOOL WEAR STATE CLASSIFICATION
# ---------------------------------------------------------------------------
def train_state_classifier(df, feature_cols):
    le = LabelEncoder()
    df = df.copy()
    df["state_enc"] = le.fit_transform(df["tool_state"])

    X_train, X_test, y_train, y_test, groups_train = group_split(df, feature_cols, "state_enc")

    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    base = xgb.XGBClassifier(objective="multi:softprob", num_class=3,
                              random_state=42, n_jobs=-1, eval_metric="mlogloss")
    cv = GroupKFold(n_splits=4)
    search = RandomizedSearchCV(
        base, param_distributions=param_dist, n_iter=15, cv=cv,
        scoring="f1_macro", random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train, groups=groups_train)
    best_model = search.best_estimator_

    preds = best_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True)

    print("\n=== Tool Wear State Classifier (XGBoost) ===")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=le.classes_))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(3), le.classes_, rotation=20)
    plt.yticks(range(3), le.classes_)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Tool Wear State — Confusion Matrix\nAccuracy={acc:.1%}, Macro-F1={f1:.3f}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/state_confusion_matrix.png")
    plt.close()

    metrics = {"accuracy": round(acc, 4), "macro_f1": round(f1, 4),
               "best_params": search.best_params_,
               "per_class": {k: v for k, v in report.items() if k in le.classes_}}
    return best_model, X_train, X_test, y_test, preds, metrics, le


# ---------------------------------------------------------------------------
# 3. SHAP INTERPRETABILITY
# ---------------------------------------------------------------------------
def run_shap_analysis(model, X_test, tag):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, max_display=12)
    plt.title(f"SHAP Feature Importance — {tag}")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/shap_summary_{tag}.png", bbox_inches="tight")
    plt.close()

    # mean |SHAP value| ranking -> top physics-driven root causes
    vals = shap_values.values
    if vals.ndim == 3:  # multiclass: average across classes
        mean_abs = np.abs(vals).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(vals).mean(axis=0)
    ranking = sorted(zip(X_test.columns, mean_abs), key=lambda x: -x[1])
    return ranking[:10]


def main():
    df, feature_cols = load_data()

    rul_model, X_train_r, X_test_r, y_test_r, preds_r, rul_metrics = train_rul_regressor(df, feature_cols)
    rul_shap_ranking = run_shap_analysis(rul_model, X_test_r.sample(min(600, len(X_test_r)), random_state=1), "RUL_Regressor")

    clf_model, X_train_c, X_test_c, y_test_c, preds_c, clf_metrics, le = train_state_classifier(df, feature_cols)
    clf_shap_ranking = run_shap_analysis(clf_model, X_test_c.sample(min(600, len(X_test_c)), random_state=1), "State_Classifier")

    summary = {
        "rul_regressor_metrics": rul_metrics,
        "rul_top_shap_drivers": rul_shap_ranking,
        "state_classifier_metrics": clf_metrics,
        "state_top_shap_drivers": clf_shap_ranking,
    }

    with open(f"{REPORT_DIR}/model_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nTop 10 SHAP drivers — RUL:")
    for feat, val in rul_shap_ranking:
        print(f"  {feat:35s} {val:.3f}")
    print("\nTop 10 SHAP drivers — Tool Wear State:")
    for feat, val in clf_shap_ranking:
        print(f"  {feat:35s} {val:.3f}")

    print(f"\nAll figures saved to {FIG_DIR}")
    print(f"Summary JSON saved to {REPORT_DIR}/model_results_summary.json")


if __name__ == "__main__":
    main()
