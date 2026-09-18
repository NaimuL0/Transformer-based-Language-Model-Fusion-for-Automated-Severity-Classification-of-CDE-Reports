"""
Training Pipeline for CDS-Fusion-V2
=============================================
Clinically-Motivated Dual-Stream Fusion Network

Complete pipeline:
  1. Data loading and preprocessing
  2. Two-stage augmentation (findings-only)
  3. Repeated Stratified K-Fold (3×5)
  4. R-Drop training with gradient accumulation
  5. Stochastic Weight Averaging (SWA)
  6. Comprehensive evaluation with CI
  7. Visualization and results export
  8. Interpretability (gate analysis, attention extraction)

Usage:
  python train_cds_fusion_v2.py

  Or with custom config:
  python train_cds_fusion_v2.py --data_path /path/to/Final_data.csv --epochs 40

Author: [Your Name]
Affiliation: Kennesaw State University / InComm Healthcare
"""

import os
import re
import json
import copy
import random
import logging
import warnings
import argparse
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

from cds_fusion_v2_model import (
    CDSFusionV2,
    CDSFusionV2Loss,
    DualStreamEchoDataset,
    FindingsAugmenter,
    augment_dataset_two_stage,
    prepare_data,
    compute_class_weights,
    get_predictions,
    setup_encoder,
    LoRALinear,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class V2Config:
    """Training configuration optimized for 718 samples with dual-stream split."""

    # ---- Data ----
    data_path: str = "/content/Final_data.csv"
    label_col: str = "label"
    text_col: str = "Medical_Report"
    label_map: Dict[str, int] = {
        "Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3,
    }
    num_classes: int = 4

    # ---- Model ----
    biobert_name: str = "dmis-lab/biobert-base-cased-v1.2"
    clinicalbert_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_findings_len: int = 256    # ~189 max tokens observed, 256 gives margin
    max_impression_len: int = 64   # ~58 max tokens observed, 64 gives margin
    lora_rank: int = 8
    lora_alpha: float = 16.0
    dropout: float = 0.25
    num_dropout_samples: int = 5
    encoder_mode: str = "unfreeze"  # "unfreeze" (recommended) or "lora"
    num_unfreeze_layers: int = 4    # Unfreeze last 4 BERT layers per encoder

    # ---- Training ----
    num_folds: int = 5
    num_repeats: int = 3           # 3×5 = 15 evaluations for stable CI
    epochs: int = 40
    batch_size: int = 8
    gradient_accumulation_steps: int = 4   # effective batch = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 10
    swa_start_epoch: int = 25

    # ---- Loss ----
    focal_weight: float = 0.6        # Primary: Focal CE for class imbalance
    ordinal_weight: float = 0.2      # Auxiliary: ordinal distance penalty
    rdrop_weight: float = 0.2
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1     # Reduced from 0.15 (was too aggressive)
    severe_boost: float = 1.5        # Extra weight for Severe class
    encoder_lr_factor: float = 0.3   # FIX v2.2: encoder_lr = lr × 0.3 (was 0.1, too low)

    # ---- Augmentation ----
    augment_train: bool = True
    on_the_fly_augment: bool = True   # Additional per-epoch augmentation

    # ---- Misc ----
    seed: int = 42
    output_dir: str = "outputs_v2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # FIXED: AMP only on GPU. On CPU it causes numerical instability
    use_amp: bool = torch.cuda.is_available()
    num_workers: int = 0 if not torch.cuda.is_available() else 2  # 0 on CPU avoids multiprocessing overhead

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__class__.__dict__.items()
                if not k.startswith('_') and not callable(v)}


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 2. TEXT PREPROCESSING
# ============================================================================

def preprocess_report(text: str) -> str:
    """
    Light cleaning of raw Medical_Report text before splitting.
    Note: Unit conversion (mm→millimeters, %→percent) is now handled
    by _linearize_findings(), so we only do whitespace and nan cleanup here.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# 3. ON-THE-FLY AUGMENTED DATASET
# ============================================================================

class OnTheFlyDualStreamDataset(DualStreamEchoDataset):
    """
    Extends DualStreamEchoDataset with on-the-fly augmentation.
    
    During training, each __getitem__ call randomly augments the FINDINGS
    portion of the text with a different augmentation each epoch.
    This creates implicit data expansion without storing augmented copies.
    
    The IMPRESSION portion is NEVER augmented.
    
    FIX v2.2: Augmentation happens on RAW findings (key-value pairs),
    then linearization converts to natural language for BERT.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        bio_tokenizer: AutoTokenizer,
        clin_tokenizer: AutoTokenizer,
        label_map_inv: Dict[int, str],
        max_findings_len: int = 256,
        max_impression_len: int = 64,
        augment: bool = False,
        augment_prob: float = 0.4,
        linearize: bool = True,
    ):
        # Don't call parent __init__ — we handle splitting differently
        self.labels = labels
        self.bio_tokenizer = bio_tokenizer
        self.clin_tokenizer = clin_tokenizer
        self.max_findings_len = max_findings_len
        self.max_impression_len = max_impression_len
        self.augment = augment
        self.label_map_inv = label_map_inv
        self.linearize = linearize

        # Store RAW findings (key-value format) for augmentation
        # and impressions separately
        self.raw_findings = []
        self.impression_texts = []
        for text in texts:
            findings, impression = self._split_report(text)
            self.raw_findings.append(findings)
            self.impression_texts.append(impression)
        
        # Pre-linearize for validation (no augmentation)
        if not augment:
            self.findings_texts = []
            for f in self.raw_findings:
                self.findings_texts.append(
                    self._linearize_findings(f) if linearize else f
                )

        # Per-class augmenters
        self.augmenters = {}
        for label_id, label_name in label_map_inv.items():
            self.augmenters[label_id] = FindingsAugmenter(
                severity_label=label_name, augment_prob=augment_prob
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        impression = self.impression_texts[idx]
        label = self.labels[idx]

        if self.augment:
            # Augment RAW findings first (key-value format)
            raw = self.raw_findings[idx]
            if label in self.augmenters:
                raw = self.augmenters[label].augment_findings(raw)
            # Then linearize the augmented findings
            findings = self._linearize_findings(raw) if self.linearize else raw
        else:
            findings = self.findings_texts[idx]

        # Tokenize
        findings_enc = self.bio_tokenizer(
            findings,
            max_length=self.max_findings_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        impression_enc = self.clin_tokenizer(
            impression,
            max_length=self.max_impression_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'bio_input_ids': findings_enc['input_ids'].squeeze(0),
            'bio_attention_mask': findings_enc['attention_mask'].squeeze(0),
            'clin_input_ids': impression_enc['input_ids'].squeeze(0),
            'clin_attention_mask': impression_enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


# ============================================================================
# 4. WEIGHTED SAMPLER FOR IMBALANCED DATA
# ============================================================================

def get_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Inverse-frequency weighted sampler for class-balanced batches."""
    counts = Counter(labels)
    class_weights = {k: 1.0 / v for k, v in counts.items()}
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(labels), replacement=True,
    )


# ============================================================================
# 5. TRAINING LOOP (R-Drop: dual forward pass)
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CDSFusionV2Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    scaler: GradScaler,
    config: V2Config,
    epoch: int,
) -> Dict[str, float]:
    """
    Train one epoch with R-Drop regularization.
    
    R-Drop runs TWO forward passes with different dropout masks per batch
    and minimizes the KL divergence between outputs. This acts as a strong
    regularizer, especially effective for small datasets (Liang et al., 2021).
    """
    model.train()
    total_loss = 0.0
    total_focal = 0.0
    total_ordinal = 0.0
    total_rdrop = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        bio_ids = batch["bio_input_ids"].to(config.device)
        bio_mask = batch["bio_attention_mask"].to(config.device)
        clin_ids = batch["clin_input_ids"].to(config.device)
        clin_mask = batch["clin_attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        with autocast(enabled=config.use_amp):
            # Forward pass 1
            outputs1 = model(bio_ids, bio_mask, clin_ids, clin_mask)
            # Forward pass 2 (different dropout masks → R-Drop)
            outputs2 = model(bio_ids, bio_mask, clin_ids, clin_mask)

            losses = criterion(outputs1, outputs2, labels)
            loss = losses["total_loss"] / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += losses["total_loss"].item()
        total_focal += losses["focal_loss"].item()
        total_ordinal += losses["ordinal_loss"].item()
        total_rdrop += losses["rdrop_loss"].item()
        num_batches += 1

    return {
        "train_loss": total_loss / max(num_batches, 1),
        "focal_loss": total_focal / max(num_batches, 1),
        "ordinal_loss": total_ordinal / max(num_batches, 1),
        "rdrop_loss": total_rdrop / max(num_batches, 1),
    }


# ============================================================================
# 6. EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: CDSFusionV2Loss,
    config: V2Config,
) -> Dict:
    """
    Evaluate with full metrics including gate value analysis.
    
    Returns comprehensive metrics + interpretability data:
    - Standard: accuracy, F1, QWK, MAE
    - Interpretability: per-class gate values (findings vs impression reliance)
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_gate_values = []
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        bio_ids = batch["bio_input_ids"].to(config.device)
        bio_mask = batch["bio_attention_mask"].to(config.device)
        clin_ids = batch["clin_input_ids"].to(config.device)
        clin_mask = batch["clin_attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        with autocast(enabled=config.use_amp):
            outputs = model(bio_ids, bio_mask, clin_ids, clin_mask)
            losses = criterion(outputs, None, labels)

        total_loss += losses["total_loss"].item()
        num_batches += 1

        probs = outputs["probs"]
        preds = probs.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

        # Collect gate values for interpretability
        if "gate_values" in outputs:
            gate = outputs["gate_values"].mean(dim=-1)  # Average across hidden dim
            all_gate_values.extend(gate.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    inv_map = {v: k for k, v in config.label_map.items()}
    target_names = [inv_map[i] for i in range(config.num_classes)]

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = 0.0

    # Gate analysis: per-class average gate value
    gate_analysis = {}
    if all_gate_values:
        gate_arr = np.array(all_gate_values)
        for cls_id, cls_name in inv_map.items():
            mask = all_labels == cls_id
            if mask.sum() > 0:
                gate_analysis[cls_name] = {
                    "mean_gate": float(gate_arr[mask].mean()),
                    "std_gate": float(gate_arr[mask].std()),
                }

    return {
        "val_loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "qwk": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
        "mae": float(np.mean(np.abs(all_preds - all_labels))),
        "roc_auc": roc_auc,
        "classification_report": classification_report(
            all_labels, all_preds, target_names=target_names,
            digits=4, output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "gate_analysis": gate_analysis,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
    }


# ============================================================================
# 7. STOCHASTIC WEIGHT AVERAGING (SWA)
# ============================================================================

class SWAModel:
    """
    Running average of model weights across late epochs.
    Leads to wider optima → better generalization on small datasets.
    """
    def __init__(self, model: nn.Module):
        self.swa_state = copy.deepcopy(model.state_dict())
        self.num_averaged = 0

    def update(self, model: nn.Module):
        state = model.state_dict()
        self.num_averaged += 1
        for key in self.swa_state:
            if state[key].is_floating_point():
                self.swa_state[key] = (
                    self.swa_state[key] * (self.num_averaged - 1) + state[key]
                ) / self.num_averaged

    def apply(self, model: nn.Module):
        model.load_state_dict(self.swa_state)


# ============================================================================
# 8. VISUALIZATION
# ============================================================================

def plot_results(fold_metrics: List[Dict], config: V2Config, output_dir: str):
    """Comprehensive visualization of training results."""

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor('#0f1419')
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

    inv_map = {v: k for k, v in config.label_map.items()}
    class_names = [inv_map[i] for i in range(config.num_classes)]
    colors = ['#4ade80', '#fbbf24', '#fb923c', '#f87171']

    def style_ax(ax, title):
        ax.set_facecolor('#0f1419')
        ax.set_title(title, color='#e2e8f0', fontsize=11, fontweight='bold', pad=10)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#2a3555')

    # 1. F1 Macro across folds
    ax1 = fig.add_subplot(gs[0, 0])
    style_ax(ax1, 'F1 Macro Across Folds')
    f1s = [m["f1_macro"] for m in fold_metrics]
    ax1.bar(range(1, len(f1s)+1), f1s, color='#3b82f6', alpha=0.8, edgecolor='#1a2238')
    ax1.axhline(y=np.mean(f1s), color='#22d3ee', linestyle='--',
                label=f'Mean: {np.mean(f1s):.3f}±{np.std(f1s):.3f}')
    ax1.fill_between(range(0, len(f1s)+2), np.mean(f1s)-np.std(f1s),
                     np.mean(f1s)+np.std(f1s), alpha=0.15, color='#22d3ee')
    ax1.set_xlabel('Fold', color='#94a3b8')
    ax1.set_ylabel('F1 Macro', color='#94a3b8')
    ax1.legend(facecolor='#1a2238', edgecolor='#2a3555', labelcolor='#e2e8f0', fontsize=8)
    ax1.set_ylim(0, 1)

    # 2. Aggregate confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    style_ax(ax2, 'Aggregate Confusion Matrix')
    total_cm = sum(m["confusion_matrix"] for m in fold_metrics)
    cm_norm = total_cm.astype(float) / total_cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names, ax=ax2,
                annot_kws={'size': 9, 'fontweight': 'bold'}, cbar_kws={'label': '%'})
    ax2.set_xlabel('Predicted', color='#94a3b8')
    ax2.set_ylabel('True', color='#94a3b8')

    # 3. Per-class F1 boxplot
    ax3 = fig.add_subplot(gs[0, 2])
    style_ax(ax3, 'Per-Class F1 Distribution')
    per_class_f1 = {cn: [] for cn in class_names}
    for m in fold_metrics:
        report = m["classification_report"]
        for cn in class_names:
            if cn in report:
                per_class_f1[cn].append(report[cn]["f1-score"])
    bp = ax3.boxplot([per_class_f1[cn] for cn in class_names],
                     labels=class_names, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color + '40')
        patch.set_edgecolor(color)
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp[element]:
            item.set_color('#94a3b8')
    ax3.set_ylabel('F1 Score', color='#94a3b8')
    ax3.set_ylim(0, 1)

    # 4. Gate analysis (NEW — V2 specific)
    ax4 = fig.add_subplot(gs[0, 3])
    style_ax(ax4, 'Gate Values (Findings vs Impression)')
    gate_means = {cn: [] for cn in class_names}
    for m in fold_metrics:
        ga = m.get("gate_analysis", {})
        for cn in class_names:
            if cn in ga:
                gate_means[cn].append(ga[cn]["mean_gate"])
    if any(gate_means.values()):
        positions = range(len(class_names))
        means = [np.mean(gate_means[cn]) if gate_means[cn] else 0 for cn in class_names]
        stds = [np.std(gate_means[cn]) if gate_means[cn] else 0 for cn in class_names]
        bars = ax4.bar(positions, means, yerr=stds, color=colors, alpha=0.8,
                       edgecolor='#1a2238', capsize=4, error_kw={'color': '#94a3b8'})
        ax4.set_xticks(positions)
        ax4.set_xticklabels(class_names, color='#94a3b8')
        ax4.set_ylabel('Mean Gate Value', color='#94a3b8')
        ax4.axhline(y=0.5, color='#94a3b8', linestyle=':', alpha=0.5)
        ax4.text(0.02, 0.95, 'g→1: relies on findings\ng→0: relies on impression',
                transform=ax4.transAxes, color='#64748b', fontsize=7, va='top')
        ax4.set_ylim(0, 1)

    # 5. Metrics summary
    ax5 = fig.add_subplot(gs[1, 0])
    style_ax(ax5, 'Aggregate Metrics (Mean ± Std)')
    metrics_keys = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted', 'qwk', 'roc_auc']
    metric_labels = ['Accuracy', 'Bal. Acc', 'F1 Macro', 'F1 Wt', 'QWK', 'ROC-AUC']
    means = [np.mean([m[k] for m in fold_metrics]) for k in metrics_keys]
    stds = [np.std([m[k] for m in fold_metrics]) for k in metrics_keys]
    bars = ax5.barh(metric_labels, means, xerr=stds, color='#3b82f6', alpha=0.8,
                    edgecolor='#1a2238', capsize=3, error_kw={'color': '#94a3b8'})
    for bar, mean, std in zip(bars, means, stds):
        ax5.text(mean + std + 0.02, bar.get_y() + bar.get_height()/2,
                f'{mean:.3f}±{std:.3f}', va='center', color='#e2e8f0', fontsize=8)
    ax5.set_xlim(0, 1.2)

    # 6. MAE distribution
    ax6 = fig.add_subplot(gs[1, 1])
    style_ax(ax6, 'Mean Absolute Error (Ordinal)')
    maes = [m["mae"] for m in fold_metrics]
    ax6.bar(range(1, len(maes)+1), maes, color='#ef4444', alpha=0.7, edgecolor='#1a2238')
    ax6.axhline(y=np.mean(maes), color='#fbbf24', linestyle='--',
               label=f'Mean: {np.mean(maes):.3f}')
    ax6.set_xlabel('Fold', color='#94a3b8')
    ax6.set_ylabel('MAE', color='#94a3b8')
    ax6.legend(facecolor='#1a2238', edgecolor='#2a3555', labelcolor='#e2e8f0', fontsize=8)

    # 7. Architecture diagram (text)
    ax7 = fig.add_subplot(gs[1, 2])
    style_ax(ax7, 'CDS-Fusion-V2 Architecture')
    ax7.axis('off')
    arch_text = [
        ("Stream 1 (BioBERT)", "Clinical Findings", "#3b82f6"),
        ("  ↓ LoRA adapters", "Last 4 layers", "#64748b"),
        ("  ↓ Findings CDA", "Attention pooling", "#64748b"),
        ("Stream 2 (ClinicalBERT)", "Impression Text", "#8b5cf6"),
        ("  ↓ LoRA adapters", "Last 4 layers", "#64748b"),
        ("  ↓ Impression CDA", "Attention pooling", "#64748b"),
        ("Cross-Stream Fusion", "Gated: g⊙F + (1-g)⊙I", "#06b6d4"),
        ("HCFA", "3-group hierarchical", "#f59e0b"),
        ("CORAL + Focal", "Ordinal classifier", "#ef4444"),
    ]
    for i, (name, desc, color) in enumerate(arch_text):
        y = 0.95 - i * 0.105
        ax7.text(0.05, y, name, fontsize=9, color=color,
                transform=ax7.transAxes, fontweight='bold')
        ax7.text(0.55, y, desc, fontsize=8, color='#94a3b8',
                transform=ax7.transAxes)

    # 8. Strategy summary
    ax8 = fig.add_subplot(gs[1, 3])
    style_ax(ax8, 'Low-Resource Strategies')
    ax8.axis('off')
    strategies = [
        ("Dual-Stream Split", "Findings ↔ Impression", "#3b82f6"),
        ("Findings-Only Augment", "No impression modification", "#06b6d4"),
        ("LoRA Adapters", "~99% param reduction", "#8b5cf6"),
        ("R-Drop", "KL divergence regularization", "#f59e0b"),
        ("CORAL + Focal Loss", "Ordinal + imbalance", "#ef4444"),
        ("Multi-Sample Dropout", "5 masks averaged", "#4ade80"),
        ("SWA", "Wider optima generalization", "#f472b6"),
        ("Repeated K-Fold", "3×5 = 15 evaluations", "#94a3b8"),
    ]
    for i, (name, desc, color) in enumerate(strategies):
        y = 0.95 - i * 0.115
        ax8.text(0.02, y, "●", fontsize=12, color=color,
                transform=ax8.transAxes, fontweight='bold')
        ax8.text(0.08, y, name, fontsize=9, color='#e2e8f0',
                transform=ax8.transAxes, fontweight='bold')
        ax8.text(0.48, y, desc, fontsize=8, color='#94a3b8',
                transform=ax8.transAxes)

    plt.savefig(os.path.join(output_dir, "cds_fusion_v2_results.png"), dpi=300,
                bbox_inches='tight', facecolor='#0f1419', edgecolor='none')
    plt.close()
    logger.info(f"Results visualization saved to {output_dir}/cds_fusion_v2_results.png")


# ============================================================================
# 9. MAIN TRAINING LOOP
# ============================================================================

def main(config: V2Config = None):
    if config is None:
        config = V2Config()

    set_seed(config.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("CDS-Fusion-V2: Clinically-Motivated Dual-Stream Training")
    logger.info("=" * 70)
    logger.info(f"Device: {config.device}")
    logger.info(f"Output: {output_dir}")

    # ---- Load Data ----
    texts, labels, label_map_inv = prepare_data(config.data_path)

    # Preprocess
    texts = [preprocess_report(t) for t in texts]
    logger.info(f"Preprocessed {len(texts)} reports")

    # Class weights
    np_labels = np.array(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(config.num_classes),
        y=np_labels,
    )
    class_weights[3] *= config.severe_boost  # Boost Severe
    cw_tensor = torch.FloatTensor(class_weights).to(config.device)
    logger.info(f"Class weights (Severe boosted ×{config.severe_boost}): "
                f"{dict(zip(config.label_map.keys(), class_weights.round(3)))}")

    # ---- Tokenizers ----
    logger.info("Loading tokenizers...")
    bio_tokenizer = AutoTokenizer.from_pretrained(config.biobert_name)
    clin_tokenizer = AutoTokenizer.from_pretrained(config.clinicalbert_name)

    # ---- Repeated Stratified K-Fold ----
    rskf = RepeatedStratifiedKFold(
        n_splits=config.num_folds,
        n_repeats=config.num_repeats,
        random_state=config.seed,
    )

    all_fold_metrics = []
    fold_count = 0

    for train_idx, val_idx in rskf.split(texts, labels):
        fold_count += 1
        repeat_num = (fold_count - 1) // config.num_folds + 1
        fold_in_repeat = (fold_count - 1) % config.num_folds + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Repeat {repeat_num}, Fold {fold_in_repeat} "
                     f"(Global {fold_count}/{config.num_folds * config.num_repeats})")
        logger.info(f"{'='*60}")

        # Split data
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        # Two-stage augmentation (findings only, uniform + oversampling)
        if config.augment_train:
            orig_counts = Counter(train_labels)
            train_texts, train_labels = augment_dataset_two_stage(
                train_texts, train_labels, label_map_inv
            )
            new_counts = Counter(train_labels)
            logger.info(f"Augmentation: {dict(orig_counts)} → {dict(new_counts)}")

        logger.info(f"Train: {len(train_texts)} | Val: {len(val_texts)}")

        # Create datasets
        train_dataset = OnTheFlyDualStreamDataset(
            train_texts, train_labels, bio_tokenizer, clin_tokenizer,
            label_map_inv=label_map_inv,
            max_findings_len=config.max_findings_len,
            max_impression_len=config.max_impression_len,
            augment=config.on_the_fly_augment,
        )
        val_dataset = OnTheFlyDualStreamDataset(
            val_texts, val_labels, bio_tokenizer, clin_tokenizer,
            label_map_inv=label_map_inv,
            max_findings_len=config.max_findings_len,
            max_impression_len=config.max_impression_len,
            augment=False,
        )

        sampler = get_weighted_sampler(train_labels)

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, sampler=sampler,
            num_workers=config.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size * 2, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
        )

        # ---- Model ----
        model = CDSFusionV2(
            biobert_name=config.biobert_name,
            clinicalbert_name=config.clinicalbert_name,
            num_classes=config.num_classes,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            dropout=config.dropout,
            num_dropout_samples=config.num_dropout_samples,
            encoder_mode=config.encoder_mode,
            num_unfreeze_layers=config.num_unfreeze_layers,
        ).to(config.device)

        # ---- Loss ----
        criterion = CDSFusionV2Loss(
            num_classes=config.num_classes,
            focal_weight=config.focal_weight,
            ordinal_weight=config.ordinal_weight,
            rdrop_weight=config.rdrop_weight,
            focal_gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            class_weights=cw_tensor,
        )

        # ---- Optimizer with differential learning rates ----
        # Encoder layers need LOWER LR to avoid catastrophic forgetting
        # Head layers (adapters, CDA, fusion, classifier) use full LR
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'biobert.encoder' in name or 'clinicalbert.encoder' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        encoder_lr = config.learning_rate * config.encoder_lr_factor
        optimizer = AdamW([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params, 'lr': config.learning_rate},
        ], weight_decay=config.weight_decay)

        logger.info(f"  Optimizer: encoder_lr={encoder_lr:.1e}, head_lr={config.learning_rate:.1e}")
        logger.info(f"  Encoder params: {len(encoder_params)}, Head params: {len(head_params)}")

        total_steps = (len(train_loader) // config.gradient_accumulation_steps) * config.epochs
        scheduler = OneCycleLR(
            optimizer, max_lr=[encoder_lr, config.learning_rate],
            total_steps=max(1, total_steps), pct_start=0.1,
            anneal_strategy='cos',
        )
        scaler = GradScaler(enabled=config.use_amp)

        # ---- SWA ----
        swa = SWAModel(model)
        swa_active = False

        # ---- Training loop ----
        best_f1 = 0.0
        patience_counter = 0
        best_val_metrics = None

        for epoch in range(1, config.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer,
                scheduler, scaler, config, epoch,
            )

            val_metrics = evaluate(model, val_loader, criterion, config)

            # SWA
            if epoch >= config.swa_start_epoch:
                if not swa_active:
                    logger.info(f"  → SWA activated at epoch {epoch}")
                    swa_active = True
                swa.update(model)

            # Logging
            gate_info = ""
            ga = val_metrics.get("gate_analysis", {})
            if ga:
                gate_vals = [ga[k]["mean_gate"] for k in ga]
                gate_info = f" Gate: {np.mean(gate_vals):.3f}"

            logger.info(
                f"  Epoch {epoch:3d}/{config.epochs} | "
                f"Loss: {train_metrics['train_loss']:.4f} "
                f"(F:{train_metrics['focal_loss']:.3f} "
                f"O:{train_metrics['ordinal_loss']:.3f} "
                f"R:{train_metrics['rdrop_loss']:.3f}) | "
                f"Val F1m: {val_metrics['f1_macro']:.4f} "
                f"QWK: {val_metrics['qwk']:.4f}{gate_info}"
            )

            # Best model tracking
            if val_metrics["f1_macro"] > best_f1:
                best_f1 = val_metrics["f1_macro"]
                patience_counter = 0
                best_val_metrics = val_metrics
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "fold": fold_count,
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "config": config.to_dict(),
                }, os.path.join(output_dir, f"best_model_fold{fold_count}.pt"))
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    logger.info(f"  Early stopping at epoch {epoch} (best F1: {best_f1:.4f})")
                    break

        # Apply SWA and compare
        if swa_active:
            swa.apply(model)
            swa_metrics = evaluate(model, val_loader, criterion, config)
            logger.info(f"  SWA: F1m={swa_metrics['f1_macro']:.4f} vs best={best_f1:.4f}")
            if swa_metrics["f1_macro"] > best_f1:
                best_val_metrics = swa_metrics
                best_f1 = swa_metrics["f1_macro"]
                logger.info("  → Using SWA model (better than best checkpoint)")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "fold": fold_count,
                    "epoch": "swa",
                    "best_f1": best_f1,
                }, os.path.join(output_dir, f"swa_model_fold{fold_count}.pt"))

        # Log fold results
        report = best_val_metrics["classification_report"]
        logger.info(f"\n  Fold {fold_count} Best (F1m={best_f1:.4f}):")
        for label_name in config.label_map.keys():
            if label_name in report:
                cls = report[label_name]
                logger.info(
                    f"    {label_name:>10s}: P={cls['precision']:.3f} "
                    f"R={cls['recall']:.3f} F1={cls['f1-score']:.3f} "
                    f"N={cls['support']}"
                )

        # Gate analysis per class
        ga = best_val_metrics.get("gate_analysis", {})
        if ga:
            logger.info("  Gate values (findings reliance):")
            for cls_name, vals in ga.items():
                logger.info(f"    {cls_name:>10s}: {vals['mean_gate']:.3f} ± {vals['std_gate']:.3f}")

        all_fold_metrics.append(best_val_metrics)

    # ================================================================
    # AGGREGATE RESULTS
    # ================================================================
    logger.info(f"\n{'='*70}")
    logger.info("AGGREGATE RESULTS (Repeated Stratified K-Fold)")
    logger.info(f"  {config.num_repeats} repeats × {config.num_folds} folds "
                f"= {len(all_fold_metrics)} evaluations")
    logger.info(f"{'='*70}")

    metric_names = ["accuracy", "balanced_accuracy", "f1_macro",
                    "f1_weighted", "qwk", "mae", "roc_auc"]
    summary = {}
    for metric in metric_names:
        values = [m[metric] for m in all_fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        ci_95 = 1.96 * std_val / np.sqrt(len(values))
        summary[metric] = {
            "mean": round(float(mean_val), 4),
            "std": round(float(std_val), 4),
            "ci_95": round(float(ci_95), 4),
            "values": [round(float(v), 4) for v in values],
        }
        logger.info(f"  {metric:>20s}: {mean_val:.4f} ± {std_val:.4f} "
                     f"(95% CI: ±{ci_95:.4f})")

    # Per-class F1
    logger.info("\n  Per-class F1 (aggregated):")
    per_class_summary = {}
    for label_name in config.label_map.keys():
        f1s = [m["classification_report"].get(label_name, {}).get("f1-score", 0)
               for m in all_fold_metrics]
        per_class_summary[label_name] = {
            "mean": round(float(np.mean(f1s)), 4),
            "std": round(float(np.std(f1s)), 4),
        }
        logger.info(f"    {label_name:>10s}: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    # Gate analysis aggregate
    logger.info("\n  Gate analysis (aggregate):")
    gate_summary = {}
    for label_name in config.label_map.keys():
        gates = [m["gate_analysis"].get(label_name, {}).get("mean_gate", 0.5)
                 for m in all_fold_metrics if m.get("gate_analysis")]
        if gates:
            gate_summary[label_name] = {
                "mean": round(float(np.mean(gates)), 4),
                "std": round(float(np.std(gates)), 4),
            }
            logger.info(f"    {label_name:>10s}: g={np.mean(gates):.3f} ± {np.std(gates):.3f}")

    summary["per_class_f1"] = per_class_summary
    summary["gate_analysis"] = gate_summary

    # Save summary
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Visualization
    plot_results(all_fold_metrics, config, output_dir)

    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Training complete!")

    return summary, all_fold_metrics


# ============================================================================
# 10. CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CDS-Fusion-V2")
    parser.add_argument("--data_path", type=str, default="Final_data.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs_v2")
    parser.add_argument("--no_augment", action="store_true")
    args = parser.parse_args()

    config = V2Config()
    config.data_path = args.data_path
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.num_folds = args.folds
    config.num_repeats = args.repeats
    config.seed = args.seed
    config.output_dir = args.output_dir
    if args.no_augment:
        config.augment_train = False
        config.on_the_fly_augment = False

    main(config)
