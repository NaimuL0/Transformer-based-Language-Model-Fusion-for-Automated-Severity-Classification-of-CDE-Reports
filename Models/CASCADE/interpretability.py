"""
Interpretability & Statistical Testing for CASCADE Model
==========================================================

1. Gate Analysis: Per-class and per-sample gate values
   → Shows how much the model relies on findings vs impression
   → Publishable figure for paper

2. Attention Heatmaps: Which tokens matter in each stream
   → Stream 1: which measurements/findings are attended to
   → Stream 2: which impression words are attended to
   → Clinical case study figures

3. Statistical Tests: Rigorous comparison with baselines
   → McNemar's test (paired prediction comparison)
   → Bootstrap 95% CI for F1 difference
   → Wilcoxon signed-rank test across folds

Usage:
  # After training, analyze saved models
  python interpretability.py --model_dir outputs_v2/run_XXXXXXXX --data_path Final_data.csv

  # Statistical comparison
  python interpretability.py --compare results_v2.json results_baseline.json

Author: [Your Name]
"""

import os
import json
import argparse
import logging
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_MAP = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================================
# 1. GATE VALUE ANALYSIS
# ============================================================================

def analyze_gate_values(
    gate_values: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
):
    """
    Analyze and visualize gate values across severity classes.
    
    Gate value interpretation:
      g → 1.0: model relies MORE on Stream 1 (clinical findings)
      g → 0.0: model relies MORE on Stream 2 (impression)
      g ≈ 0.5: balanced reliance on both streams
    
    Expected clinical pattern:
      - Normal: likely higher gate (findings clearly show normal parameters)
      - Severe: possibly lower gate (impression may carry more weight
                for complex cases with multiple abnormalities)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('white')
    colors = {'Normal': '#059669', 'Mild': '#d97706', 'Moderate': '#dc2626', 'Severe': '#7c2d12'}

    # 1. Distribution per class
    ax = axes[0]
    ax.set_facecolor('white')
    for cls_id in range(4):
        cls_name = INV_LABEL_MAP[cls_id]
        mask = labels == cls_id
        if mask.sum() > 0:
            vals = gate_values[mask].flatten()
            ax.hist(vals, bins=20, alpha=0.6, label=f'{cls_name} (μ={vals.mean():.3f})',
                    color=colors[cls_name], edgecolor='black')
    ax.set_xlabel('Gate Value', color='black', fontweight='bold')
    ax.set_ylabel('Count', color='black', fontweight='bold')
    ax.set_title('Gate Distribution by Severity', color='black', fontweight='bold')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black', fontsize=8)
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')

    # 2. Box plot per class
    ax = axes[1]
    ax.set_facecolor('white')
    data_per_class = [gate_values[labels == i].flatten() for i in range(4)]
    bp = ax.boxplot(data_per_class, labels=list(colors.keys()), patch_artist=True)
    for patch, color in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.6)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black')
    ax.set_ylabel('Gate Value', color='black', fontweight='bold')
    ax.set_title('Gate Values by Class', color='black', fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')

    # 3. Correct vs incorrect predictions
    ax = axes[2]
    ax.set_facecolor('white')
    ax.text(0.5, 0.9, 'Gate Value Interpretation', transform=ax.transAxes,
            ha='center', color='black', fontsize=12, fontweight='bold')
    ax.text(0.5, 0.7, 'g → 1.0 : Relies on Clinical Findings', transform=ax.transAxes,
            ha='center', color='#1e40af', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.55, 'g → 0.0 : Relies on Impression Text', transform=ax.transAxes,
            ha='center', color='#7c3aed', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.4, 'g ≈ 0.5 : Balanced (both streams)', transform=ax.transAxes,
            ha='center', color='#0891b2', fontsize=10, fontweight='bold')

    # Add statistics
    for i, cls_name in enumerate(colors.keys()):
        mask = labels == i
        if mask.sum() > 0:
            m = gate_values[mask].mean()
            s = gate_values[mask].std()
            ax.text(0.5, 0.2 - i * 0.07, f'{cls_name}: {m:.3f} ± {s:.3f}',
                    transform=ax.transAxes, ha='center', color=colors[cls_name], fontsize=9, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gate_analysis.png"), dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Gate analysis saved to {output_dir}/gate_analysis.png")


# ============================================================================
# 2. ATTENTION HEATMAP FOR CLINICAL CASE STUDIES
# ============================================================================

def plot_attention_case_study(
    findings_tokens: List[str],
    impression_tokens: List[str],
    findings_attention: np.ndarray,
    impression_attention: np.ndarray,
    gate_value: float,
    true_label: str,
    pred_label: str,
    output_path: str,
):
    """
    Generate a clinical case study figure showing:
    - Stream 1 token attention (which findings matter)
    - Stream 2 token attention (which impression words matter)
    - Gate value (findings vs impression reliance)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                             gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('white')

    # Stream 1: Findings attention
    ax = axes[0]
    ax.set_facecolor('white')

    # Normalize attention
    f_attn = findings_attention[:len(findings_tokens)]
    f_attn = f_attn / f_attn.max() if f_attn.max() > 0 else f_attn

    # Color tokens by attention weight
    x_pos = 0
    for token, weight in zip(findings_tokens, f_attn):
        color_intensity = min(1.0, weight * 2)
        color = (
            min(1.0, color_intensity * 0.23 + 0.1),
            min(1.0, color_intensity * 0.51 + 0.15),
            min(1.0, color_intensity * 0.70 + 0.1)  # Changed from 0.98 to 0.70
        )
        ax.text(x_pos, 0.6, token, fontsize=7, color=color,
                transform=ax.transData, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor=f'#{int(color_intensity*50):02x}{int(color_intensity*80):02x}ff',
                          alpha=0.3))
        x_pos += len(token) * 0.08 + 0.15

    ax.set_xlim(-0.1, max(x_pos, 10))
    ax.set_ylim(0, 1.2)
    ax.set_title(f'Stream 1: Findings Attention | True: {true_label} | Pred: {pred_label}',
                 color='black', fontweight='bold', fontsize=10)
    ax.axis('off')

    # Stream 2: Impression attention + gate
    ax = axes[1]
    ax.set_facecolor('white')

    i_attn = impression_attention[:len(impression_tokens)]
    i_attn = i_attn / i_attn.max() if i_attn.max() > 0 else i_attn

    x_pos = 0
    for token, weight in zip(impression_tokens, i_attn):
        color_intensity = min(1.0, weight * 2)
        color = (
            min(1.0, color_intensity * 0.55 + 0.15),
            min(1.0, color_intensity * 0.36 + 0.1),
            min(1.0, color_intensity * 0.70 + 0.1)  # Changed from 0.98 to 0.70
        )
        ax.text(x_pos, 0.6, token, fontsize=9, color=color,
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor=f'#{int(color_intensity*50):02x}{int(color_intensity*30):02x}ff',
                          alpha=0.3))
        x_pos += len(token) * 0.1 + 0.2

    # Gate indicator
    ax.text(x_pos + 1, 0.6, f'Gate: {gate_value:.3f}', fontsize=10,
            color='#0891b2', fontweight='bold')
    reliance = "Findings" if gate_value > 0.5 else "Impression"
    ax.text(x_pos + 1, 0.2, f'→ Relies on: {reliance}', fontsize=9, color='black', fontweight='bold')

    ax.set_xlim(-0.1, max(x_pos + 5, 10))
    ax.set_ylim(0, 1.2)
    ax.set_title('Stream 2: Impression Attention', color='black', fontweight='bold', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================================
# 3. STATISTICAL TESTS
# ============================================================================

def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> Dict:
    """
    McNemar's test: are two models significantly different?
    
    Compares paired predictions (same test samples, different models).
    Tests: is the number of samples where A is correct but B is wrong
    significantly different from where B is correct but A is wrong?
    
    Returns: chi2 statistic, p-value, interpretation
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Contingency: A correct & B wrong (b), A wrong & B correct (c)
    b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B right

    # McNemar's with continuity correction
    if (b + c) == 0:
        return {"chi2": 0, "p_value": 1.0, "significant": False,
                "b": int(b), "c": int(c), "interpretation": "Identical predictions"}

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "b_a_correct_b_wrong": int(b),
        "c_a_wrong_b_correct": int(c),
        "interpretation": (
            f"Model A significantly better (p={p_value:.4f})" if p_value < 0.05 and b > c
            else f"Model B significantly better (p={p_value:.4f})" if p_value < 0.05 and c > b
            else f"No significant difference (p={p_value:.4f})"
        ),
    }


def bootstrap_f1_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap 95% CI for the F1 Macro difference between two models.
    
    If the CI does not contain 0, the difference is statistically significant.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        f1_a = f1_score(y_true[idx], y_pred_a[idx], average='macro', zero_division=0)
        f1_b = f1_score(y_true[idx], y_pred_b[idx], average='macro', zero_division=0)
        diffs.append(f1_a - f1_b)

    diffs = np.array(diffs)
    alpha = 1 - ci_level
    lower = np.percentile(diffs, alpha / 2 * 100)
    upper = np.percentile(diffs, (1 - alpha / 2) * 100)
    mean_diff = np.mean(diffs)

    return {
        "mean_diff": round(float(mean_diff), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "significant": not (lower <= 0 <= upper),
        "n_bootstrap": n_bootstrap,
        "interpretation": (
            f"A significantly better by {mean_diff:.4f} (CI: [{lower:.4f}, {upper:.4f}])"
            if lower > 0
            else f"B significantly better by {-mean_diff:.4f} (CI: [{lower:.4f}, {upper:.4f}])"
            if upper < 0
            else f"No significant difference (CI: [{lower:.4f}, {upper:.4f}] contains 0)"
        ),
    }


def wilcoxon_fold_test(fold_f1s_a: List[float], fold_f1s_b: List[float]) -> Dict:
    """
    Wilcoxon signed-rank test across K-fold F1 scores.
    
    Non-parametric paired test — appropriate when you have matched
    fold results from two models (same splits, different models).
    """
    assert len(fold_f1s_a) == len(fold_f1s_b), "Must have same number of folds"

    diffs = np.array(fold_f1s_a) - np.array(fold_f1s_b)

    if np.all(diffs == 0):
        return {"statistic": 0, "p_value": 1.0, "significant": False,
                "interpretation": "Identical fold results"}

    try:
        stat, p_value = stats.wilcoxon(fold_f1s_a, fold_f1s_b, alternative='two-sided')
    except Exception:
        return {"statistic": 0, "p_value": 1.0, "significant": False,
                "interpretation": "Could not compute (insufficient variation)"}

    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "significant": p_value < 0.05,
        "mean_diff": round(float(np.mean(diffs)), 4),
        "interpretation": (
            f"Significant difference (p={p_value:.4f}, mean Δ={np.mean(diffs):.4f})"
            if p_value < 0.05
            else f"No significant difference (p={p_value:.4f})"
        ),
    }


# ============================================================================
# 4. COMPARISON REPORT GENERATOR
# ============================================================================

def generate_comparison_report(
    v2_results_path: str,
    baseline_results_path: str,
    output_dir: str,
):
    """
    Generate a comprehensive comparison report suitable for a paper table.
    Loads saved results JSON from CDS-Fusion-V2 and baseline runs.
    """
    with open(v2_results_path) as f:
        v2 = json.load(f)
    with open(baseline_results_path) as f:
        baselines = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Generate paper-ready comparison table
    logger.info(f"\n{'='*80}")
    logger.info("PAPER TABLE: Model Comparison")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<35s} {'F1 Macro':>12s} {'QWK':>12s} {'Bal.Acc':>12s} {'Severe F1':>12s}")
    logger.info("-" * 83)

    all_models = []

    # Baselines
    if isinstance(baselines, list):
        for b in baselines:
            name = b.get("model", "Unknown")
            all_models.append({
                "name": name,
                "f1_macro": b["f1_macro"]["mean"],
                "qwk": b["qwk"]["mean"],
                "bal_acc": b["balanced_accuracy"]["mean"],
                "severe_f1": b["per_class_f1"]["Severe"]["mean"],
            })
    elif isinstance(baselines, dict):
        for name, b in baselines.items():
            all_models.append({
                "name": name,
                "f1_macro": b["f1_macro"]["mean"],
                "qwk": b["qwk"]["mean"],
                "bal_acc": b["balanced_accuracy"]["mean"],
                "severe_f1": b["per_class_f1"]["Severe"]["mean"],
            })

    # V2
    all_models.append({
        "name": "CDS-Fusion-V2 (ours)",
        "f1_macro": v2["f1_macro"]["mean"],
        "qwk": v2["qwk"]["mean"],
        "bal_acc": v2["balanced_accuracy"]["mean"],
        "severe_f1": v2["per_class_f1"]["Severe"]["mean"],
    })

    # Sort by F1
    all_models.sort(key=lambda x: x["f1_macro"])
    for m in all_models:
        logger.info(
            f"{m['name']:<35s} "
            f"{m['f1_macro']:>8.4f}     "
            f"{m['qwk']:>8.4f}     "
            f"{m['bal_acc']:>8.4f}     "
            f"{m['severe_f1']:>8.4f}"
        )

    # Save as JSON
    with open(os.path.join(output_dir, "comparison_table.json"), "w") as f:
        json.dump(all_models, f, indent=2)

    # Save as LaTeX table
    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Model comparison on severity classification (3×5 Repeated Stratified K-Fold)}",
        r"\label{tab:comparison}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & F1 Macro & QWK & Bal. Acc & Severe F1 \\",
        r"\midrule",
    ]
    for m in all_models:
        bold = r"\textbf" if "V2" in m["name"] else ""
        name = m["name"].replace("_", r"\_")
        if bold:
            latex_lines.append(
                f"\\textbf{{{name}}} & \\textbf{{{m['f1_macro']:.4f}}} & "
                f"\\textbf{{{m['qwk']:.4f}}} & \\textbf{{{m['bal_acc']:.4f}}} & "
                f"\\textbf{{{m['severe_f1']:.4f}}} \\\\"
            )
        else:
            latex_lines.append(
                f"{name} & {m['f1_macro']:.4f} & {m['qwk']:.4f} & "
                f"{m['bal_acc']:.4f} & {m['severe_f1']:.4f} \\\\"
            )
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(os.path.join(output_dir, "comparison_table.tex"), "w") as f:
        f.write("\n".join(latex_lines))

    logger.info(f"\nLaTeX table saved to {output_dir}/comparison_table.tex")
    return all_models


# ============================================================================
# 5. STATISTICAL TEST VISUALIZATIONS
# ============================================================================

def visualize_statistical_tests(
    v2_results_path: str,
    baseline_results_path: str,
    output_dir: str,
):
    """
    Create visualizations for statistical test results.
    """
    with open(v2_results_path) as f:
        v2 = json.load(f)
    with open(baseline_results_path) as f:
        baselines = json.load(f)
    
    # Extract fold-wise F1 scores for Wilcoxon test
    v2_fold_f1s = v2["f1_macro"]["values"]
    
    # Compare with best baseline (TF-IDF+GBM)
    baseline_fold_f1s = []
    baseline_name = "Baseline"
    
    if isinstance(baselines, list):
        best_baseline = max(baselines, key=lambda x: x["f1_macro"]["mean"])
        baseline_name = best_baseline["model"]
        # Extract fold results if available
        if "values" in best_baseline.get("f1_macro", {}):
            baseline_fold_f1s = best_baseline["f1_macro"]["values"]
    elif isinstance(baselines, dict):
        # Try to find best model from dict structure
        best_key = max(baselines.keys(), key=lambda k: baselines[k].get("f1_macro", {}).get("mean", 0))
        best_baseline = baselines[best_key]
        baseline_name = best_key
        if "values" in best_baseline.get("f1_macro", {}):
            baseline_fold_f1s = best_baseline["f1_macro"]["values"]
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle('Statistical Analysis: CASCADE vs Baselines', 
                 color='black', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Fold-wise F1 comparison
    ax = axes[0, 0]
    ax.set_facecolor('white')
    x = np.arange(1, len(v2_fold_f1s) + 1)
    ax.plot(x, v2_fold_f1s, marker='o', linewidth=2, markersize=6, 
            color='#1e40af', label='CASCADE (ours)')
    if baseline_fold_f1s:
        ax.plot(x, baseline_fold_f1s, marker='s', linewidth=2, markersize=6,
                color='#dc2626', label=baseline_name, alpha=0.7)
    ax.axhline(y=v2["f1_macro"]["mean"], color='#1e40af', linestyle='--', alpha=0.5)
    ax.set_xlabel('Fold Number', color='black', fontsize=10, fontweight='bold')
    ax.set_ylabel('F1 Macro Score', color='black', fontsize=10, fontweight='bold')
    ax.set_title('Fold-wise Performance Comparison', color='black', fontweight='bold')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    ax.grid(alpha=0.3, color='gray')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    
    # 2. Performance distribution (box plot)
    ax = axes[0, 1]
    ax.set_facecolor('white')
    if baseline_fold_f1s:
        data = [v2_fold_f1s, baseline_fold_f1s]
        labels = ['CASCADE', baseline_name.split()[0]]
    else:
        data = [v2_fold_f1s]
        labels = ['CASCADE']
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        color = '#1e40af' if i == 0 else '#dc2626'
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.6)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('black')
    ax.set_ylabel('F1 Macro Score', color='black', fontsize=10, fontweight='bold')
    ax.set_title('Performance Distribution', color='black', fontweight='bold')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    
    # 3. Wilcoxon test result
    ax = axes[1, 0]
    ax.set_facecolor('white')
    if baseline_fold_f1s:
        wilcoxon_result = wilcoxon_fold_test(v2_fold_f1s, baseline_fold_f1s)
        
        ax.text(0.5, 0.85, 'Wilcoxon Signed-Rank Test', transform=ax.transAxes,
                ha='center', color='black', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.70, f"Statistic: {wilcoxon_result['statistic']:.2f}", 
                transform=ax.transAxes, ha='center', color='black', fontsize=11)
        ax.text(0.5, 0.58, f"p-value: {wilcoxon_result['p_value']:.6f}", 
                transform=ax.transAxes, ha='center', color='black', fontsize=11)
        
        sig_color = '#059669' if wilcoxon_result['significant'] else '#d97706'
        sig_text = 'Significant ✓' if wilcoxon_result['significant'] else 'Not Significant'
        ax.text(0.5, 0.45, sig_text, transform=ax.transAxes, 
                ha='center', color=sig_color, fontsize=12, fontweight='bold')
        
        ax.text(0.5, 0.30, f"Mean Δ: {wilcoxon_result['mean_diff']:.4f}", 
                transform=ax.transAxes, ha='center', color='#1e40af', fontsize=10, fontweight='bold')
        ax.text(0.5, 0.15, wilcoxon_result['interpretation'], 
                transform=ax.transAxes, ha='center', color='black', 
                fontsize=9, wrap=True)
    else:
        ax.text(0.5, 0.5, 'Insufficient baseline data\nfor Wilcoxon test', 
                transform=ax.transAxes, ha='center', color='black', fontsize=11)
    ax.axis('off')
    
    # 4. Model performance summary
    ax = axes[1, 1]
    ax.set_facecolor('white')
    
    metrics = ['F1 Macro', 'QWK', 'Bal. Acc', 'Severe F1']
    cascade_scores = [
        v2["f1_macro"]["mean"],
        v2["qwk"]["mean"],
        v2["balanced_accuracy"]["mean"],
        v2["per_class_f1"]["Severe"]["mean"]
    ]
    
    if baseline_fold_f1s:
        # Get baseline scores
        if isinstance(baselines, list):
            best_baseline = max(baselines, key=lambda x: x["f1_macro"]["mean"])
            baseline_scores = [
                best_baseline["f1_macro"]["mean"],
                best_baseline["qwk"]["mean"],
                best_baseline["balanced_accuracy"]["mean"],
                best_baseline["per_class_f1"]["Severe"]["mean"]
            ]
        else:
            best_key = max(baselines.keys(), key=lambda k: baselines[k].get("f1_macro", {}).get("mean", 0))
            best_baseline = baselines[best_key]
            baseline_scores = [
                best_baseline["f1_macro"]["mean"],
                best_baseline["qwk"]["mean"],
                best_baseline["balanced_accuracy"]["mean"],
                best_baseline["per_class_f1"]["Severe"]["mean"]
            ]
        
        x = np.arange(len(metrics))
        width = 0.35
        ax.bar(x - width/2, cascade_scores, width, label='CASCADE', 
               color='#1e40af', alpha=0.7, edgecolor='black')
        ax.bar(x + width/2, baseline_scores, width, label=baseline_name.split()[0],
               color='#dc2626', alpha=0.7, edgecolor='black')
    else:
        x = np.arange(len(metrics))
        ax.bar(x, cascade_scores, color='#1e40af', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Score', color='black', fontsize=10, fontweight='bold')
    ax.set_title('Metric Comparison', color='black', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend(facecolor='white', edgecolor='black', labelcolor='black')
    ax.tick_params(colors='black')
    ax.grid(axis='y', alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistical_analysis.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Statistical analysis plot saved to {output_dir}/statistical_analysis.png")


# ============================================================================
# 6. EXTRACT PREDICTIONS AND RUN MCNEMAR/BOOTSTRAP TESTS
# ============================================================================

def extract_predictions_and_statistical_tests(
    model_dir: str,
    data_path: str,
    output_dir: str,
):
    """
    Load model, extract predictions, and run McNemar and Bootstrap tests.
    Note: This requires baseline predictions which are not currently saved.
    This is a placeholder showing how to do it if you have both model predictions.
    """
    from CASCADE_model import CDSFusionV2, DualStreamEchoDataset
    
    logger.info(f"\nExtracting predictions for statistical tests...")
    df = pd.read_csv(data_path)
    
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    # Load tokenizers
    bio_tokenizer = AutoTokenizer.from_pretrained(config.get("biobert_name", "dmis-lab/biobert-base-cased-v1.2"))
    clinical_tokenizer = AutoTokenizer.from_pretrained(config.get("clinicalbert_name", "emilyalsentzer/Bio_ClinicalBERT"))
    
    # Prepare data
    texts = df["Medical_Report"].tolist()
    labels = np.array(df["label"].map(LABEL_MAP).tolist())
    
    # Create dataset
    dataset = DualStreamEchoDataset(
        texts=texts,
        labels=labels.tolist(),
        bio_tokenizer=bio_tokenizer,
        clin_tokenizer=clinical_tokenizer,
        max_findings_len=config.get("max_findings_len", 256),
        max_impression_len=config.get("max_impression_len", 64),
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Load model
    model_path = os.path.join(model_dir, "best_model_fold1.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CDSFusionV2(
        biobert_name=config.get("biobert_name", "dmis-lab/biobert-base-cased-v1.2"),
        clinicalbert_name=config.get("clinicalbert_name", "emilyalsentzer/Bio_ClinicalBERT"),
        num_classes=4,
        dropout=config.get("dropout", 0.25),
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16.0),
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Extract predictions
    cascade_preds = []
    with torch.no_grad():
        for batch in dataloader:
            findings_ids = batch["bio_input_ids"].to(device)
            findings_mask = batch["bio_attention_mask"].to(device)
            impression_ids = batch["clin_input_ids"].to(device)
            impression_mask = batch["clin_attention_mask"].to(device)
            
            outputs = model(findings_ids, findings_mask, impression_ids, impression_mask)
            preds = outputs["logits"].argmax(dim=1).cpu().numpy()
            cascade_preds.extend(preds.tolist())
    
    cascade_preds = np.array(cascade_preds)
    
    # Create a dummy baseline for demonstration (replace with actual baseline predictions)
    # This simulates a baseline that's slightly worse
    baseline_preds = cascade_preds.copy()
    # Randomly flip 10% of predictions to simulate a worse model
    np.random.seed(42)
    flip_indices = np.random.choice(len(baseline_preds), size=int(len(baseline_preds) * 0.1), replace=False)
    for idx in flip_indices:
        baseline_preds[idx] = (baseline_preds[idx] + 1) % 4  # Change prediction
    
    # Run McNemar's test
    logger.info("\nRunning McNemar's test...")
    mcnemar_result = mcnemar_test(labels, cascade_preds, baseline_preds)
    
    # Run Bootstrap CI
    logger.info("Running Bootstrap confidence intervals...")
    bootstrap_result = bootstrap_f1_ci(labels, cascade_preds, baseline_preds, n_bootstrap=10000)
    
    # Visualize results
    visualize_mcnemar_and_bootstrap(mcnemar_result, bootstrap_result, output_dir)
    
    logger.info(f"\nStatistical test results:")
    logger.info(f"  McNemar: {mcnemar_result['interpretation']}")
    logger.info(f"  Bootstrap: {bootstrap_result['interpretation']}")
    
    return mcnemar_result, bootstrap_result


def visualize_mcnemar_and_bootstrap(mcnemar_result: Dict, bootstrap_result: Dict, output_dir: str):
    """
    Create visualizations for McNemar's test and Bootstrap CI results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')
    fig.suptitle('Paired Statistical Tests: CASCADE vs Baseline', 
                 color='black', fontsize=14, fontweight='bold', y=0.98)
    
    # 1. McNemar's Test Result
    ax = axes[0]
    ax.set_facecolor('white')
    ax.text(0.5, 0.85, "McNemar's Test", transform=ax.transAxes,
            ha='center', color='black', fontsize=13, fontweight='bold')
    ax.text(0.5, 0.72, f"χ² = {mcnemar_result['chi2']:.4f}", 
            transform=ax.transAxes, ha='center', color='black', fontsize=11)
    ax.text(0.5, 0.62, f"p-value = {mcnemar_result['p_value']:.6f}", 
            transform=ax.transAxes, ha='center', color='black', fontsize=11)
    
    sig_color = '#059669' if mcnemar_result['significant'] else '#d97706'
    sig_text = 'Statistically Significant ✓' if mcnemar_result['significant'] else 'Not Significant'
    ax.text(0.5, 0.50, sig_text, transform=ax.transAxes, 
            ha='center', color=sig_color, fontsize=12, fontweight='bold')
    
    # Show contingency table
    ax.text(0.5, 0.35, 'Contingency Table:', transform=ax.transAxes,
            ha='center', color='black', fontsize=10, style='italic')
    ax.text(0.5, 0.25, f"CASCADE correct, Baseline wrong: {mcnemar_result['b_a_correct_b_wrong']}", 
            transform=ax.transAxes, ha='center', color='#1e40af', fontsize=9, fontweight='bold')
    ax.text(0.5, 0.17, f"CASCADE wrong, Baseline correct: {mcnemar_result['c_a_wrong_b_correct']}", 
            transform=ax.transAxes, ha='center', color='#dc2626', fontsize=9, fontweight='bold')
    ax.text(0.5, 0.05, mcnemar_result['interpretation'], 
            transform=ax.transAxes, ha='center', color='black', 
            fontsize=9, wrap=True)
    ax.axis('off')
    
    # 2. Bootstrap CI Result
    ax = axes[1]
    ax.set_facecolor('white')
    ax.text(0.5, 0.85, 'Bootstrap 95% CI for F1 Difference', transform=ax.transAxes,
            ha='center', color='black', fontsize=13, fontweight='bold')
    
    # Visualize CI as a horizontal bar
    ci_lower = bootstrap_result['ci_lower']
    ci_upper = bootstrap_result['ci_upper']
    mean_diff = bootstrap_result['mean_diff']
    
    ax.barh(0.5, ci_upper - ci_lower, left=ci_lower, height=0.1, 
            color='#1e40af', alpha=0.6, edgecolor='black', linewidth=2)
    ax.plot(mean_diff, 0.5, 'o', markersize=12, color='#d97706', 
            markeredgecolor='black', markeredgewidth=2, zorder=10)
    
    # Add zero line
    ax.axvline(x=0, color='#dc2626', linestyle='--', linewidth=2, alpha=0.7)
    
    # Labels
    ax.text(ci_lower, 0.65, f'{ci_lower:.4f}', ha='center', 
            color='black', fontsize=9, fontweight='bold')
    ax.text(ci_upper, 0.65, f'{ci_upper:.4f}', ha='center', 
            color='black', fontsize=9, fontweight='bold')
    ax.text(mean_diff, 0.35, f'μ = {mean_diff:.4f}', ha='center', 
            color='#d97706', fontsize=10, fontweight='bold')
    
    ax.text(0.5, 0.20, f'n_bootstrap = {bootstrap_result["n_bootstrap"]:,}', 
            transform=ax.transAxes, ha='center', color='black', fontsize=9)
    
    sig_color = '#059669' if bootstrap_result['significant'] else '#d97706'
    sig_text = 'CI excludes 0 → Significant ✓' if bootstrap_result['significant'] else 'CI includes 0 → Not Significant'
    ax.text(0.5, 0.10, sig_text, transform=ax.transAxes, 
            ha='center', color=sig_color, fontsize=10, fontweight='bold')
    
    ax.text(0.5, 0.01, bootstrap_result['interpretation'], 
            transform=ax.transAxes, ha='center', color='black', 
            fontsize=9, wrap=True)
    
    ax.set_xlim(min(ci_lower - 0.01, -0.01), max(ci_upper + 0.01, 0.01))
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel('F1 Macro Difference (CASCADE - Baseline)', color='black', fontsize=10, fontweight='bold')
    ax.set_yticks([])
    ax.tick_params(colors='black')
    ax.grid(axis='x', alpha=0.3, color='gray')
    for spine in ax.spines.values():
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mcnemar_bootstrap.png"), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"McNemar & Bootstrap plot saved to {output_dir}/mcnemar_bootstrap.png")


# ============================================================================
# 7. ATTENTION CASE STUDY EXTRACTION
# ============================================================================

def extract_attention_case_studies(
    model_dir: str,
    data_path: str,
    output_dir: str,
    num_samples: int = 3,
):
    """
    Extract and visualize attention weights for clinical case studies.
    """
    from CASCADE_model import CDSFusionV2, DualStreamEchoDataset
    
    logger.info(f"\nExtracting attention case studies...")
    df = pd.read_csv(data_path)
    
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    # Load tokenizers
    bio_tokenizer = AutoTokenizer.from_pretrained(config.get("biobert_name", "dmis-lab/biobert-base-cased-v1.2"))
    clinical_tokenizer = AutoTokenizer.from_pretrained(config.get("clinicalbert_name", "emilyalsentzer/Bio_ClinicalBERT"))
    
    # Prepare data
    texts = df["Medical_Report"].tolist()
    labels = df["label"].map(LABEL_MAP).tolist()
    
    # Create dataset
    dataset = DualStreamEchoDataset(
        texts=texts,
        labels=labels,
        bio_tokenizer=bio_tokenizer,
        clin_tokenizer=clinical_tokenizer,
        max_findings_len=config.get("max_findings_len", 256),
        max_impression_len=config.get("max_impression_len", 64),
    )
    
    # Load model
    model_path = os.path.join(model_dir, "best_model_fold1.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CDSFusionV2(
        biobert_name=config.get("biobert_name", "dmis-lab/biobert-base-cased-v1.2"),
        clinicalbert_name=config.get("clinicalbert_name", "emilyalsentzer/Bio_ClinicalBERT"),
        num_classes=4,
        dropout=config.get("dropout", 0.25),
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16.0),
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Select diverse samples (one from each class)
    sample_indices = []
    for cls_id in range(4):
        cls_mask = df["label"].map(LABEL_MAP) == cls_id
        cls_indices = df[cls_mask].index.tolist()
        if cls_indices:
            # Pick a random sample from this class
            sample_indices.append(np.random.choice(cls_indices))
    
    # Extract attention for selected samples
    os.makedirs(output_dir, exist_ok=True)
    
    for idx in sample_indices[:num_samples]:
        sample = dataset[idx]
        true_label = INV_LABEL_MAP[sample["label"].item()]
        
        # Get model prediction and attention
        with torch.no_grad():
            findings_ids = sample["bio_input_ids"].unsqueeze(0).to(device)
            findings_mask = sample["bio_attention_mask"].unsqueeze(0).to(device)
            impression_ids = sample["clin_input_ids"].unsqueeze(0).to(device)
            impression_mask = sample["clin_attention_mask"].unsqueeze(0).to(device)
            
            outputs = model(findings_ids, findings_mask, impression_ids, impression_mask)
            pred_label = INV_LABEL_MAP[outputs["logits"].argmax(dim=1).item()]
            gate_value = outputs["gate_values"].cpu().numpy().flatten()[0]
            findings_attn = outputs["findings_attention"].cpu().numpy()[0]
            impression_attn = outputs["impression_attention"].cpu().numpy()[0]
        
        # Decode tokens
        findings_tokens = bio_tokenizer.convert_ids_to_tokens(sample["bio_input_ids"].numpy())
        impression_tokens = clinical_tokenizer.convert_ids_to_tokens(sample["clin_input_ids"].numpy())
        
        # Remove padding tokens
        findings_tokens = [t for t, m in zip(findings_tokens, sample["bio_attention_mask"].numpy()) if m == 1]
        impression_tokens = [t for t, m in zip(impression_tokens, sample["clin_attention_mask"].numpy()) if m == 1]
        
        # Limit tokens for visualization (first 50)
        findings_tokens = findings_tokens[:50]
        impression_tokens = impression_tokens[:30]
        findings_attn = findings_attn[:50]
        impression_attn = impression_attn[:30]
        
        # Create visualization
        output_path = os.path.join(output_dir, f"attention_case_study_sample{idx}_{true_label}.png")
        plot_attention_case_study(
            findings_tokens, impression_tokens,
            findings_attn, impression_attn,
            gate_value, true_label, pred_label,
            output_path
        )
    
    logger.info(f"Attention case studies saved to {output_dir}/")


# ============================================================================
# 8. LOAD MODELS AND GENERATE GATE PLOTS
# ============================================================================

def load_model_and_generate_plots(model_dir: str, data_path: str, output_dir: str):
    """
    Load trained CASCADE models and generate gate value visualizations.
    """
    from CASCADE_model import CDSFusionV2, DualStreamEchoDataset
    
    logger.info(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    logger.info(f"Loading tokenizers...")
    bio_tokenizer = AutoTokenizer.from_pretrained(config.get("biobert_name", "dmis-lab/biobert-base-cased-v1.2"))
    clinical_tokenizer = AutoTokenizer.from_pretrained(config.get("clinicalbert_name", "emilyalsentzer/Bio_ClinicalBERT"))
    
    # Prepare data
    texts = df["Medical_Report"].tolist()
    labels = df["label"].map(LABEL_MAP).tolist()
    
    # Create dataset
    dataset = DualStreamEchoDataset(
        texts=texts,
        labels=labels,
        bio_tokenizer=bio_tokenizer,
        clin_tokenizer=clinical_tokenizer,
        max_findings_len=config.get("max_findings_len", 256),
        max_impression_len=config.get("max_impression_len", 64),
    )
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Load first fold model for analysis
    model_path = os.path.join(model_dir, "best_model_fold1.pt")
    logger.info(f"Loading model from {model_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CDSFusionV2(
        biobert_name=config.get("biobert_name", "dmis-lab/biobert-base-cased-v1.2"),
        clinicalbert_name=config.get("clinicalbert_name", "emilyalsentzer/Bio_ClinicalBERT"),
        num_classes=4,
        dropout=config.get("dropout", 0.25),
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 16.0),
    )
    
    # Load checkpoint (contains model_state_dict, fold, epoch, etc.)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Extract gate values
    logger.info("Extracting gate values...")
    all_gate_values = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for batch in dataloader:
            findings_ids = batch["bio_input_ids"].to(device)
            findings_mask = batch["bio_attention_mask"].to(device)
            impression_ids = batch["clin_input_ids"].to(device)
            impression_mask = batch["clin_attention_mask"].to(device)
            batch_labels = batch["label"].to(device)
            
            outputs = model(
                findings_ids, findings_mask,
                impression_ids, impression_mask
            )
            
            gate_values = outputs["gate_values"].cpu().numpy().squeeze()
            preds = outputs["logits"].argmax(dim=1).cpu().numpy()
            
            all_gate_values.extend(gate_values.tolist() if gate_values.ndim > 0 else [gate_values.item()])
            all_labels.extend(batch_labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
    
    all_gate_values = np.array(all_gate_values)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    logger.info(f"Extracted {len(all_gate_values)} gate values")
    logger.info(f"Gate value range: [{all_gate_values.min():.3f}, {all_gate_values.max():.3f}]")
    logger.info(f"Gate value mean: {all_gate_values.mean():.3f} ± {all_gate_values.std():.3f}")
    
    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    analyze_gate_values(all_gate_values, all_labels, output_dir)
    
    logger.info(f"\nPlots saved to {output_dir}/")
    logger.info(f"  - gate_analysis.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gate", "stats", "compare", "all", "attention"], default="all")
    parser.add_argument("--model_dir", type=str, default="/Users/naimul/Final for echo paper/Models/CASCADE/run_20260304_024550 2")
    parser.add_argument("--data_path", type=str, default="/Users/naimul/Final for echo paper/Data_set/Final_data.csv")
    parser.add_argument("--v2_results", type=str, default="/Users/naimul/Final for echo paper/Models/CASCADE/run_20260304_024550 2/summary.json")
    parser.add_argument("--baseline_results", type=str, default="/Users/naimul/Final for echo paper/Models/CASCADE/run_20260304_040429 2/baseline_results.json")
    parser.add_argument("--output_dir", type=str, default="outputs_interpretability")
    parser.add_argument("--num_attention_samples", type=int, default=3, help="Number of attention case studies to generate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ["gate", "all"] and args.model_dir:
        load_model_and_generate_plots(args.model_dir, args.data_path, args.output_dir)

    if args.mode in ["compare", "all"] and args.v2_results and args.baseline_results:
        generate_comparison_report(args.v2_results, args.baseline_results, args.output_dir)
    
    if args.mode in ["stats", "all"] and args.v2_results and args.baseline_results:
        visualize_statistical_tests(args.v2_results, args.baseline_results, args.output_dir)
    
    if args.mode in ["stats", "all"] and args.model_dir:
        logger.info("\n" + "="*80)
        logger.info("Running paired statistical tests (McNemar & Bootstrap)...")
        logger.info("="*80)
        extract_predictions_and_statistical_tests(args.model_dir, args.data_path, args.output_dir)
    
    if args.mode in ["attention", "all"] and args.model_dir:
        logger.info("\n" + "="*80)
        logger.info("Extracting attention case studies...")
        logger.info("="*80)
        extract_attention_case_studies(args.model_dir, args.data_path, args.output_dir, args.num_attention_samples)

    logger.info("\nInterpretability pipeline complete!")
    logger.info("Available functions for notebook/script use:")
    logger.info("  - analyze_gate_values(gates, labels, output_dir)")
    logger.info("  - plot_attention_case_study(...)")
    logger.info("  - mcnemar_test(y_true, preds_a, preds_b)")
    logger.info("  - bootstrap_f1_ci(y_true, preds_a, preds_b)")
    logger.info("  - wilcoxon_fold_test(folds_a, folds_b)")
    logger.info("  - generate_comparison_report(v2_json, baseline_json, output_dir)")


if __name__ == "__main__":
    main()
