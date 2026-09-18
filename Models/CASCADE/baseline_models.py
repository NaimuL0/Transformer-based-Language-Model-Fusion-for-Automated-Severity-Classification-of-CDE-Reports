"""
Baseline Models for CDS-Fusion-V2 Comparison (Tier 1)
======================================================

These baselines answer the reviewer question:
  "Why do you need a dual-stream transformer? Couldn't simpler models work?"

Baselines:
  1A. TF-IDF + Gradient Boosting (text)
  1B. TF-IDF + XGBoost (text)
  1C. TF-IDF + Random Forest (text)
  1D. TF-IDF + Logistic Regression (text)
  2A. Structured features + XGBoost (tabular)
  2B. Structured features + Random Forest (tabular)

Each baseline uses Repeated Stratified K-Fold (3×5) with same seed
as CDS-Fusion-V2 for fair comparison.

Usage:
  python baseline_models.py --data_path Final_data.csv
  python baseline_models.py --data_path Final_data.csv --features_path extracted_medical_reports_only_features.xlsx

Author: [Your Name]
"""

import os
import re
import json
import argparse
import logging
import warnings
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    cohen_kappa_score, accuracy_score, balanced_accuracy_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
SEED = 42


# ============================================================================
# DATA LOADING
# ============================================================================

def load_text_data(csv_path: str) -> Tuple[List[str], List[str], List[int]]:
    """
    Load Final_data.csv and create three text variants:
    - full_text: complete Medical_Report (with Impression)
    - findings_text: everything BEFORE Impression
    - impression_text: only the Impression
    """
    df = pd.read_csv(csv_path)
    labels = [LABEL_MAP[l] for l in df['label'].tolist()]

    full_texts = []
    findings_texts = []

    for text in df['Medical_Report']:
        text = str(text)
        text_clean = re.sub(r"\bnan\b", "not measured", text, flags=re.IGNORECASE)
        full_texts.append(text_clean)

        parts = text_clean.rsplit('Impression:', 1)
        findings_texts.append(parts[0].strip().rstrip(',').strip())

    return full_texts, findings_texts, labels


def load_structured_features(xlsx_path: str) -> Tuple[np.ndarray, List[int]]:
    """
    Load structured features from xlsx.
    Handles mixed types: extracts numeric values from measurement strings,
    one-hot encodes categorical fields, drops constant/empty columns.
    """
    df = pd.read_excel(xlsx_path)

    labels = [LABEL_MAP[l] for l in df['label'].tolist()]

    # Separate feature columns
    drop_cols = ['Cardiac_Class', 'label', 'Impression']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Process each column
    processed = {}
    for col in feature_cols:
        series = df[col].astype(str).str.strip()

        # Try numeric extraction (e.g., "56 mm" → 56, "41%" → 41)
        numeric = series.str.extract(r'(\d+\.?\d*)')[0].astype(float)

        if numeric.notna().sum() > len(df) * 0.3:
            # Enough numeric values — use as numeric feature
            processed[col] = numeric.fillna(numeric.median())
        else:
            # Categorical — check unique values
            unique = series.replace(['nan', 'N/A', 'not measured', ''], 'missing').unique()
            if len(unique) <= 1:
                continue  # Constant column — skip
            if len(unique) <= 10:
                # One-hot encode
                dummies = pd.get_dummies(
                    series.replace(['nan', 'N/A', 'not measured', ''], 'missing'),
                    prefix=col
                )
                for dummy_col in dummies.columns:
                    processed[dummy_col] = dummies[dummy_col].values.astype(float)
            else:
                # Too many categories — skip (e.g., LV_Wall_Motion free text)
                continue

    features_df = pd.DataFrame(processed)
    logger.info(f"Structured features: {features_df.shape[1]} features from {len(feature_cols)} columns")

    return features_df.values, labels


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_predictions(y_true, y_pred, y_probs=None) -> Dict:
    """Compute all metrics matching CDS-Fusion-V2 evaluation."""
    target_names = [INV_LABEL_MAP[i] for i in range(4)]

    roc_auc = 0.0
    if y_probs is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except Exception:
            pass

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": float(np.mean(np.abs(np.array(y_pred) - np.array(y_true)))),
        "roc_auc": roc_auc,
        "classification_report": classification_report(
            y_true, y_pred, target_names=target_names,
            digits=4, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


# ============================================================================
# BASELINE RUNNERS
# ============================================================================

def run_tfidf_baseline(
    texts: List[str],
    labels: List[int],
    model_name: str,
    model_class,
    model_params: dict,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> Dict:
    """Run TF-IDF + classifier baseline with Repeated Stratified K-Fold."""

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(texts, labels), 1):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = np.array([labels[i] for i in train_idx])
        val_labels = np.array([labels[i] for i in val_idx])

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                sublinear_tf=True, min_df=2)
        X_train = tfidf.fit_transform(train_texts)
        X_val = tfidf.transform(val_texts)

        # Train
        clf = model_class(**model_params)
        clf.fit(X_train, train_labels)

        # Predict
        y_pred = clf.predict(X_val)
        y_probs = None
        if hasattr(clf, 'predict_proba'):
            y_probs = clf.predict_proba(X_val)

        metrics = evaluate_predictions(val_labels, y_pred, y_probs)
        fold_metrics.append(metrics)

    return aggregate_results(model_name, fold_metrics)


def run_structured_baseline(
    features: np.ndarray,
    labels: List[int],
    model_name: str,
    model_class,
    model_params: dict,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> Dict:
    """Run structured features + classifier baseline."""

    labels_arr = np.array(labels)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(rskf.split(features, labels_arr), 1):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels_arr[train_idx], labels_arr[val_idx]

        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train
        clf = model_class(**model_params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_probs = clf.predict_proba(X_val) if hasattr(clf, 'predict_proba') else None

        metrics = evaluate_predictions(y_val, y_pred, y_probs)
        fold_metrics.append(metrics)

    return aggregate_results(model_name, fold_metrics)


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_results(model_name: str, fold_metrics: List[Dict]) -> Dict:
    """Aggregate fold metrics into summary with CI."""
    metric_keys = ["accuracy", "balanced_accuracy", "f1_macro",
                   "f1_weighted", "qwk", "mae", "roc_auc"]

    summary = {"model": model_name, "n_folds": len(fold_metrics)}

    for key in metric_keys:
        values = [m[key] for m in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_95 = 1.96 * std_val / np.sqrt(len(values))
        summary[key] = {
            "mean": round(float(mean_val), 4),
            "std": round(float(std_val), 4),
            "ci_95": round(float(ci_95), 4),
        }

    # Per-class F1
    per_class = {}
    for label_name in LABEL_MAP.keys():
        f1s = [m["classification_report"].get(label_name, {}).get("f1-score", 0)
               for m in fold_metrics]
        per_class[label_name] = {
            "mean": round(float(np.mean(f1s)), 4),
            "std": round(float(np.std(f1s)), 4),
        }
    summary["per_class_f1"] = per_class

    # Aggregate confusion matrix
    summary["confusion_matrix"] = sum(m["confusion_matrix"] for m in fold_metrics).tolist()

    return summary


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_baseline_comparison(all_results: List[Dict], output_dir: str):
    """Compare all baselines side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0f1419')

    model_names = [r["model"] for r in all_results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    for ax in axes:
        ax.set_facecolor('#0f1419')
        ax.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#2a3555')

    # 1. F1 Macro comparison
    ax = axes[0]
    means = [r["f1_macro"]["mean"] for r in all_results]
    stds = [r["f1_macro"]["std"] for r in all_results]
    bars = ax.barh(model_names, means, xerr=stds, color=colors, alpha=0.8,
                   edgecolor='#1a2238', capsize=3)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.01, bar.get_y() + bar.get_height()/2,
                f'{mean:.3f}', va='center', color='#e2e8f0', fontsize=8)
    ax.set_title('F1 Macro', color='#e2e8f0', fontweight='bold')
    ax.set_xlim(0, 1.1)

    # 2. QWK comparison
    ax = axes[1]
    means = [r["qwk"]["mean"] for r in all_results]
    stds = [r["qwk"]["std"] for r in all_results]
    bars = ax.barh(model_names, means, xerr=stds, color=colors, alpha=0.8,
                   edgecolor='#1a2238', capsize=3)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(mean + std + 0.01, bar.get_y() + bar.get_height()/2,
                f'{mean:.3f}', va='center', color='#e2e8f0', fontsize=8)
    ax.set_title('QWK (Ordinal)', color='#e2e8f0', fontweight='bold')
    ax.set_xlim(0, 1.1)

    # 3. Per-class F1 for Severe (hardest class)
    ax = axes[2]
    severe_means = [r["per_class_f1"]["Severe"]["mean"] for r in all_results]
    severe_stds = [r["per_class_f1"]["Severe"]["std"] for r in all_results]
    bars = ax.barh(model_names, severe_means, xerr=severe_stds, color='#f87171',
                   alpha=0.8, edgecolor='#1a2238', capsize=3)
    for bar, mean, std in zip(bars, severe_means, severe_stds):
        ax.text(mean + std + 0.01, bar.get_y() + bar.get_height()/2,
                f'{mean:.3f}', va='center', color='#e2e8f0', fontsize=8)
    ax.set_title('F1 Severe (Minority)', color='#e2e8f0', fontweight='bold')
    ax.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "baseline_comparison.png"), dpi=300,
                bbox_inches='tight', facecolor='#0f1419')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--data_path", type=str, default="Final_data.csv")
    parser.add_argument("--features_path", type=str, default=None,
                        help="Path to structured features xlsx (optional)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="outputs_baselines")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BASELINE MODELS — CDS-Fusion-V2 Comparison")
    logger.info("=" * 70)

    # Load text data
    full_texts, findings_texts, labels = load_text_data(args.data_path)
    logger.info(f"Loaded {len(full_texts)} records, distribution: {dict(Counter(labels))}")

    all_results = []

    # ---- TEXT BASELINES ----

    # 1A. TF-IDF + GBM (full text)
    logger.info("\n--- TF-IDF + GBM (full text) ---")
    result = run_tfidf_baseline(
        full_texts, labels, "TF-IDF+GBM (full)",
        GradientBoostingClassifier,
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "random_state": SEED},
        args.folds, args.repeats,
    )
    all_results.append(result)
    logger.info(f"  F1 Macro: {result['f1_macro']['mean']:.4f} ± {result['f1_macro']['std']:.4f}")

    # 1B. TF-IDF + GBM (findings only — matches Stream 1 input)
    logger.info("\n--- TF-IDF + GBM (findings only) ---")
    result = run_tfidf_baseline(
        findings_texts, labels, "TF-IDF+GBM (findings)",
        GradientBoostingClassifier,
        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "random_state": SEED},
        args.folds, args.repeats,
    )
    all_results.append(result)
    logger.info(f"  F1 Macro: {result['f1_macro']['mean']:.4f} ± {result['f1_macro']['std']:.4f}")

    # 1C. TF-IDF + Random Forest (full text)
    logger.info("\n--- TF-IDF + RF (full text) ---")
    result = run_tfidf_baseline(
        full_texts, labels, "TF-IDF+RF (full)",
        RandomForestClassifier,
        {"n_estimators": 300, "max_depth": 15, "class_weight": "balanced", "random_state": SEED},
        args.folds, args.repeats,
    )
    all_results.append(result)
    logger.info(f"  F1 Macro: {result['f1_macro']['mean']:.4f} ± {result['f1_macro']['std']:.4f}")

    # 1D. TF-IDF + Logistic Regression (full text)
    logger.info("\n--- TF-IDF + LogReg (full text) ---")
    result = run_tfidf_baseline(
        full_texts, labels, "TF-IDF+LogReg (full)",
        LogisticRegression,
        {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "random_state": SEED},
        args.folds, args.repeats,
    )
    all_results.append(result)
    logger.info(f"  F1 Macro: {result['f1_macro']['mean']:.4f} ± {result['f1_macro']['std']:.4f}")

    # 1E. TF-IDF + XGBoost (if available)
    if HAS_XGBOOST:
        logger.info("\n--- TF-IDF + XGBoost (full text) ---")
        result = run_tfidf_baseline(
            full_texts, labels, "TF-IDF+XGB (full)",
            XGBClassifier,
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
             "use_label_encoder": False, "eval_metric": "mlogloss", "random_state": SEED},
            args.folds, args.repeats,
        )
        all_results.append(result)
        logger.info(f"  F1 Macro: {result['f1_macro']['mean']:.4f} ± {result['f1_macro']['std']:.4f}")

    # ---- STRUCTURED FEATURE BASELINES ----
    if args.features_path and os.path.exists(args.features_path):
        features, struct_labels = load_structured_features(args.features_path)

        if HAS_XGBOOST:
            logger.info("\n--- Structured + XGBoost ---")
            result = run_structured_baseline(
                features, struct_labels, "Structured+XGB",
                XGBClassifier,
                {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                 "use_label_encoder": False, "eval_metric": "mlogloss", "random_state": SEED},
                args.folds, args.repeats,
            )
            all_results.append(result)
            logger.info(f"  F1 Macro: {result['f1_macro']['mean']:.4f} ± {result['f1_macro']['std']:.4f}")

        logger.info("\n--- Structured + RF ---")
        result = run_structured_baseline(
            features, struct_labels, "Structured+RF",
            RandomForestClassifier,
            {"n_estimators": 300, "max_depth": 15, "class_weight": "balanced", "random_state": SEED},
            args.folds, args.repeats,
        )
        all_results.append(result)
        logger.info(f"  F1 Macro: {result['f1_macro']['mean']:.4f} ± {result['f1_macro']['std']:.4f}")

    # ---- SUMMARY TABLE ----
    logger.info(f"\n{'='*70}")
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Model':<30s} {'F1 Macro':>12s} {'QWK':>12s} {'Bal.Acc':>12s} {'Severe F1':>12s}")
    logger.info("-" * 78)
    for r in all_results:
        logger.info(
            f"{r['model']:<30s} "
            f"{r['f1_macro']['mean']:.4f}±{r['f1_macro']['std']:.4f} "
            f"{r['qwk']['mean']:.4f}±{r['qwk']['std']:.4f} "
            f"{r['balanced_accuracy']['mean']:.4f}±{r['balanced_accuracy']['std']:.4f} "
            f"{r['per_class_f1']['Severe']['mean']:.4f}±{r['per_class_f1']['Severe']['std']:.4f}"
        )

    # Save
    with open(os.path.join(output_dir, "baseline_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    plot_baseline_comparison(all_results, output_dir)

    logger.info(f"\nResults saved to: {output_dir}")
    return all_results


if __name__ == "__main__":
    main()
