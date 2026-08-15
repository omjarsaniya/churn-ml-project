"""
Train the churn prediction pipeline.

Run from anywhere:
    python src/train.py
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    fbeta_score,
)

from paths import RAW_DATA_PATH, PIPELINE_PATH, REPORTS_DIR
from preprocess import load_data, clean_data, split_data
from pipeline import create_pipeline


def pick_threshold_by_fbeta(y_true, y_prob, beta=2.0):
    """
    Scan candidate thresholds and pick the one that maximizes F-beta.

    beta=2 weights recall twice as heavily as precision, which matches the
    business framing in the README: a missed churner (false negative) costs
    more than an unnecessary retention offer (false positive).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta)
        for t in thresholds
    ]
    best_idx = int(np.argmax(scores))
    return thresholds[best_idx], scores[best_idx]


def main():
    # 1. Load + clean
    df = load_data(RAW_DATA_PATH)
    df = clean_data(df)
    X, y = split_data(df)

    # 2. Cross-validation — gives a mean +/- std instead of one lucky/unlucky split
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pipeline = create_pipeline(X)
    cv_results = cross_validate(
        cv_pipeline, X, y, cv=cv,
        scoring=["recall", "roc_auc"],
        n_jobs=-1,
    )
    print("=== 5-Fold Cross-Validation (full dataset) ===")
    print(f"Recall : {cv_results['test_recall'].mean():.3f} +/- {cv_results['test_recall'].std():.3f}")
    print(f"ROC-AUC: {cv_results['test_roc_auc'].mean():.3f} +/- {cv_results['test_roc_auc'].std():.3f}")

    # 3. Hold out a test set for the threshold analysis + final report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = create_pipeline(X)
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred_default = pipeline.predict(X_test)

    print("\n=== Default Threshold (0.5) — Holdout Test Set ===")
    print(classification_report(y_test, y_pred_default))
    print("ROC AUC:", round(roc_auc_score(y_test, y_prob), 3))
    print(confusion_matrix(y_test, y_pred_default))

    # 4. Pick a threshold from the precision-recall curve instead of guessing
    best_threshold, best_fbeta = pick_threshold_by_fbeta(y_test, y_prob, beta=2.0)
    y_pred_tuned = (y_prob >= best_threshold).astype(int)

    print(f"\n=== Tuned Threshold ({best_threshold:.2f}, chosen by max F2) — Holdout Test Set ===")
    print(classification_report(y_test, y_pred_tuned))
    print(confusion_matrix(y_test, y_pred_tuned))

    # 5. Save the precision-recall curve as a figure for the README/notebook
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions, label="Precision-Recall curve")
    plt.scatter(
        [recalls[np.argmin(np.abs(pr_thresholds - best_threshold))]],
        [precisions[np.argmin(np.abs(pr_thresholds - best_threshold))]],
        color="red", zorder=5,
        label=f"Chosen threshold = {best_threshold:.2f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Churn Model")
    plt.legend()
    plt.grid(alpha=0.3)
    fig_path = REPORTS_DIR / "precision_recall_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved precision-recall curve to {fig_path}")

    # 6. Refit on the FULL dataset for the deployed model (more data = better model),
    #    now that the holdout set has already told us how it generalizes.
    final_pipeline = create_pipeline(X)
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, PIPELINE_PATH)
    print(f"\nFinal pipeline (trained on full data) saved to {PIPELINE_PATH}")
    print(f"Recommended serving threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    main()
