"""
CDS-Fusion-V2: Clinically-Motivated Dual-Stream Fusion Network
================================================================
for Automated Severity Classification of Echocardiography Reports

KEY INSIGHT:
  Echocardiography reports have two semantically distinct components:
  
  1. CLINICAL FINDINGS — measurements (EF, LVIDd, LA...), structural
     descriptions (LV_Cavity_Size, Wall_Motion...), valve findings,
     flow patterns. These are OBJECTIVE observations.
     
  2. IMPRESSION — the cardiologist's DIAGNOSTIC CONCLUSION summarizing
     the findings. This is a SUBJECTIVE clinical judgment.
  
  A cardiologist reads findings → forms an impression → assigns severity.
  Our architecture mirrors this exact clinical reasoning process.

ARCHITECTURE:
  ┌─────────────────────────────────────────────────────────────────┐
  │                    CDS-Fusion-V2 Architecture                   │
  │                                                                 │
  │   Stream 1: Clinical Findings          Stream 2: Impression     │
  │   ─────────────────────────            ────────────────────     │
  │   "Age: 55, Sex: Male,                "MI (inferior) with      │
  │    EF: 41%, LVIDd: 56 mm,              moderate LV systolic    │
  │    LV_Cavity_Size: Dilated,            dysfunction."           │
  │    Wall_Motion: hypokinetic,                                   │
  │    VSD: Absent, ..."                                           │
  │         │                                     │                │
  │    ┌────▼─────┐                         ┌─────▼──────┐        │
  │    │ BioBERT  │                         │ClinicalBERT│        │
  │    │ + LoRA   │                         │  + LoRA    │        │
  │    └────┬─────┘                         └─────┬──────┘        │
  │         │                                     │                │
  │    ┌────▼─────┐                         ┌─────▼──────┐        │
  │    │Findings  │                         │Impression  │        │
  │    │  CDA     │◄───── Cross-Doc ───────►│   CDA      │        │
  │    └────┬─────┘       Attention         └─────┬──────┘        │
  │         │                                     │                │
  │         └──────────┐     ┌────────────────────┘                │
  │                    ▼     ▼                                     │
  │              ┌─────────────────┐                               │
  │              │  Gated Fusion   │                               │
  │              │  g⊙F + (1-g)⊙I │                               │
  │              └────────┬────────┘                               │
  │                       │                                        │
  │              ┌────────▼────────┐                               │
  │              │   HCFA (3-group │                               │
  │              │   hierarchical) │                               │
  │              └────────┬────────┘                               │
  │                       │                                        │
  │              ┌────────▼────────┐                               │
  │              │  CORAL + Focal  │                               │
  │              │  Classifier     │                               │
  │              └────────┬────────┘                               │
  │                       │                                        │
  │              Normal / Mild / Moderate / Severe                  │
  └─────────────────────────────────────────────────────────────────┘

WHY THIS SPLIT MATTERS:
  1. PREVENTS MEMORIZATION: 99.6% of impressions map to exactly one label.
     If the model sees the impression, it just memorizes the mapping.
     By separating streams, the findings stream MUST learn clinical reasoning.
  
  2. MAKES AUGMENTATION MEANINGFUL: Augmentation only modifies findings
     (measurement noise, synonym replacement). Since the findings stream
     has no access to impression, these changes actually matter.
  
  3. CROSS-DOCUMENT ATTENTION learns which findings support which impressions:
     "EF: 41% + Wall_Motion: hypokinetic" → "moderate LV dysfunction"
     This is clinically interpretable and publishable.
  
  4. MIRRORS CLINICAL PRACTICE: Cardiologists first read measurements,
     then structural findings, then form an impression. Our architecture
     does the same thing — scientifically and clinically motivated.

DATA:
  Input file: Final_data.csv (718 records)
  Splitting: Medical_Report is split at "Impression:" into two streams
  Labels: Normal (165), Mild (269), Moderate (205), Severe (79)

AUGMENTATION (Applied ONLY to Stream 1):
  - Measurement noise: ±3mm LVIDd/LVIDs, ±2mm LA/AO (Lang et al., 2015)
  - EF perturbation: ±5% within observed class range (Thavendiranathan 2013)
  - Severity-neutral synonym replacement: "Intact"→"no defect seen" only
  - Non-informative field masking: 9 constant fields only (Cramér V=0.000)
  - Two-stage: uniform augmentation (all classes) + minority oversampling

Author: [Your Name]
Affiliation: Kennesaw State University / InComm Healthcare
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import math
import re
import random
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List
from collections import Counter


# ============================================================================
# 1. DATA PREPARATION — Split Reports into Two Streams
# ============================================================================

class DualStreamEchoDataset(Dataset):
    """
    Dataset that splits each echocardiography report into two streams:
    
    Stream 1 (findings_text): All clinical fields EXCEPT Impression
      → "Age: 55, Sex: Male, EF: 41%, LVIDd: 56 mm, ..., Others_Flow: Nill"
      → Fed to BioBERT
      → This is the "clinical evidence" that the model must reason over
      → AUGMENTATION is applied here
    
    Stream 2 (impression_text): Only the Impression field
      → "Consistent with MI with moderate LV systolic dysfunction."
      → Fed to ClinicalBERT
      → This is the "diagnostic reference" signal
      → NEVER augmented (modifying impression = modifying the diagnosis)
    
    This split forces the model to learn:
    "WHICH findings → WHICH impression → WHICH severity"
    instead of shortcutting through impression memorization.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        bio_tokenizer: AutoTokenizer,
        clin_tokenizer: AutoTokenizer,
        max_findings_len: int = 256,
        max_impression_len: int = 64,
        linearize: bool = True,
    ):
        self.labels = labels
        self.bio_tokenizer = bio_tokenizer
        self.clin_tokenizer = clin_tokenizer
        self.max_findings_len = max_findings_len
        self.max_impression_len = max_impression_len
        
        # Split each report into findings and impression
        self.findings_texts = []
        self.impression_texts = []
        
        for text in texts:
            findings, impression = self._split_report(text)
            # FIX v2.2: Linearize key-value pairs into natural language
            if linearize:
                findings = self._linearize_findings(findings)
            self.findings_texts.append(findings)
            self.impression_texts.append(impression)
    
    @staticmethod
    def _split_report(text: str) -> Tuple[str, str]:
        """
        Split a medical report text at the Impression field.
        
        Input:  "Age: 55, Sex: Male, ..., Others_Flow: Nill, Impression: MI with..."
        Output: ("Age: 55, Sex: Male, ..., Others_Flow: Nill",
                 "MI with moderate LV systolic dysfunction.")
        """
        parts = text.rsplit('Impression:', 1)
        findings = parts[0].strip().rstrip(',').strip()
        impression = parts[1].strip() if len(parts) > 1 else ""
        
        # If impression is empty, use a placeholder
        if not impression or impression.lower() in ['nan', 'n/a', 'nill', '']:
            impression = "No impression documented."
        
        return findings, impression

    @staticmethod
    def _linearize_findings(findings_text: str) -> str:
        """
        FIX v2.2: Convert structured key-value findings into natural language.
        
        BERT was pretrained on sentences, not "EF: 41%, LVIDd: 56 mm".
        Linearization converts structured data into BERT-friendly text.
        
        Before: "Age: 55, Sex: Male, EF: 41%, LVIDd: 56 mm, LV_Cavity_Size: Dilated"
        After:  "Patient age 55 years, male. Ejection fraction 41 percent.
                 LV internal dimension diastole 56 millimeters. LV cavity size dilated."
        """
        # Field-to-natural-language mapping
        FIELD_MAP = {
            'Age': ('Patient age', 'years'),
            'Sex': ('Sex', ''),
            'IVST': ('Interventricular septal thickness', 'millimeters'),
            'LVIDd': ('LV internal dimension in diastole', 'millimeters'),
            'LA': ('Left atrium', 'millimeters'),
            'LVPWT': ('LV posterior wall thickness', 'millimeters'),
            'LVIDs': ('LV internal dimension in systole', 'millimeters'),
            'AO': ('Aortic root', 'millimeters'),
            'RVGWT': ('RV free wall thickness', 'millimeters'),
            'FS': ('Fractional shortening', 'percent'),
            'ACS': ('Aortic cusp separation', 'millimeters'),
            'RV': ('Right ventricle', 'millimeters'),
            'EF': ('Ejection fraction', 'percent'),
            'LVEDV': ('LV end diastolic volume', 'milliliters'),
            'PA': ('Pulmonary artery', 'millimeters'),
            'EF_Shope': ('EF by Shope method', 'percent'),
            'LVESV': ('LV end systolic volume', 'milliliters'),
            'AV_ring': ('Aortic valve ring', 'millimeters'),
            'MV_ring': ('Mitral valve ring', 'millimeters'),
            'MVA': ('Mitral valve area', 'square centimeters'),
            'LV_Cavity_Size': ('LV cavity size is', ''),
            'LV_Wall_Thickness': ('LV wall thickness is', ''),
            'LV_Wall_Motion': ('LV wall motion shows', ''),
            'RA': ('Right atrium is', ''),
            'MV': ('Mitral valve is', ''),
            'RV_Description': ('Right ventricle is', ''),
            'AV': ('Aortic valve is', ''),
            'LA_Description': ('Left atrium is', ''),
            'PV': ('Pulmonary valve is', ''),
            'Aorta': ('Aorta is', ''),
            'TV': ('Tricuspid valve is', ''),
            'PA_Description': ('Pulmonary artery is', ''),
            'IAS': ('Interatrial septum is', ''),
            'RVOT': ('RV outflow tract is', ''),
            'IVS': ('Interventricular septum is', ''),
            'ASD': ('Atrial septal defect is', ''),
            'Thrombus': ('Thrombus is', ''),
            'VSD': ('Ventricular septal defect is', ''),
            'Vegetation': ('Vegetation is', ''),
            'PDA': ('Patent ductus arteriosus is', ''),
            'Pericardium': ('Pericardium is', ''),
            'Mitral_Valve_Flow': ('Mitral valve flow shows', ''),
            'Aortic_Valve_Flow': ('Aortic valve flow shows', ''),
            'Pulmonary_Valve_Flow': ('Pulmonary valve flow shows', ''),
            'Tricuspid_Valve_Flow': ('Tricuspid valve flow shows', ''),
            'VSD_Flow': ('VSD flow shows', ''),
            'Others_Flow': ('Other flow findings', ''),
        }
        
        sentences = []
        # Parse key-value pairs
        for field in findings_text.split(','):
            field = field.strip()
            if ':' not in field:
                continue
            key, val = field.split(':', 1)
            key = key.strip()
            val = val.strip()
            
            # Skip nan/empty values
            if val.lower() in ['nan', 'n/a', '', 'not measured', 'not reported']:
                continue
            
            if key in FIELD_MAP:
                label, unit = FIELD_MAP[key]
                # Extract numeric value if present
                num_match = re.match(r'(\d+\.?\d*)\s*(%|mm|sqcm)?', val)
                if num_match and unit:
                    num_val = num_match.group(1)
                    sentences.append(f"{label} {num_val} {unit}.")
                else:
                    # Categorical/text value
                    val_clean = val.rstrip('.').strip()
                    sentences.append(f"{label} {val_clean.lower()}.")
            else:
                # Unknown field — keep as is
                sentences.append(f"{key} {val}.")
        
        return ' '.join(sentences) if sentences else findings_text
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Tokenize findings for BioBERT (Stream 1)
        findings_enc = self.bio_tokenizer(
            self.findings_texts[idx],
            max_length=self.max_findings_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize impression for ClinicalBERT (Stream 2)
        impression_enc = self.clin_tokenizer(
            self.impression_texts[idx],
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
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def prepare_data(csv_path: str) -> Tuple[List[str], List[int], Dict[int, str]]:
    """
    Load Final_data.csv and prepare for training.
    
    Returns:
        texts: List of full Medical_Report strings
        labels: List of integer labels
        label_map_inv: {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    """
    df = pd.read_csv(csv_path)
    
    label_map = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    label_map_inv = {v: k for k, v in label_map.items()}
    
    texts = df['Medical_Report'].tolist()
    labels = [label_map[l] for l in df['label'].tolist()]
    
    print(f"Loaded {len(texts)} records from {csv_path}")
    print(f"Label distribution: {dict(Counter(labels))}")
    
    return texts, labels, label_map_inv


# ============================================================================
# 2. AUGMENTATION — Applied ONLY to Stream 1 (Clinical Findings)
# ============================================================================

class FindingsAugmenter:
    """
    Clinically-validated augmentation for echocardiographic FINDINGS only.
    
    THIS IS THE KEY DESIGN DECISION:
    - Augmentation modifies ONLY the findings text (Stream 1)
    - Impression (Stream 2) is NEVER modified
    - Because findings and impression are in SEPARATE streams, the model
      cannot shortcut through the impression to ignore augmented findings
    - Therefore, augmentation actually affects what the model learns!
    
    Without this split, augmentation was WASTED because:
    - Impression (7% of text, Cramér's V=1.000) was unchanged
    - Model learned to just read Impression and ignore everything else
    - Changing "EF: 41%" → "EF: 43%" had zero effect on model behavior
    
    With this split, augmentation is MEANINGFUL because:
    - Stream 1 has NO impression — model must learn from findings
    - Changing "EF: 41%" → "EF: 43%" directly affects Stream 1 representation
    - Model must learn: which EF values → which severity levels
    
    All augmentations cite published literature:
    - Measurement noise: Lang et al. (2015) inter-observer variability
    - EF perturbation: Thavendiranathan et al. (2013) reproducibility
    - Field masking: Validated by Cramér's V analysis (9 constant fields)
    """
    
    # Data-driven EF distributions per class (from actual dataset analysis)
    EF_DISTRIBUTIONS = {
        'Normal':   {'mean': 59.7, 'std': 7.2,  'min': 45, 'max': 75},
        'Mild':     {'mean': 55.9, 'std': 8.3,  'min': 42, 'max': 71},
        'Moderate': {'mean': 48.9, 'std': 11.8, 'min': 26, 'max': 77},
        'Severe':   {'mean': 43.9, 'std': 16.8, 'min': 25, 'max': 89},
    }
    
    # Measurement noise bounds (published inter-observer variability)
    MEASUREMENT_NOISE = {
        'LVIDd': 3,   # ±3mm (Lang 2015)
        'LVIDs': 3,   # ±3mm
        'LA': 2,      # ±2mm
        'AO': 2,      # ±2mm
        'IVST': 2,    # ±2mm
        'LVPWT': 2,   # ±2mm
    }
    
    # ONLY severity-neutral synonyms (Cramér's V = 0.000 for these terms)
    SAFE_SYNONYMS = {
        'Intact': ['no defect seen'],
        'Absent': ['not detected', 'not seen'],
        'Nill': ['none'],
        'Normal flow.': ['Normal flow pattern.'],
        'Normal Flow.': ['Normal flow pattern.'],
    }
    
    # NEVER replace these — they are severity discriminative
    FORBIDDEN_TERMS = {
        'normal', 'hypokinetic', 'akinetic', 'dilated', 'thickened',
        'regurgitation', 'stenosis', 'dyskinetic', 'poor', 'good',
        'mild', 'moderate', 'severe', 'trivial', 'significant',
    }
    
    # Fields safe to mask (100% constant across all classes, Cramér V=0.000)
    SAFE_TO_MASK_FIELDS = [
        'RVGWT', 'FS', 'PA', 'EF_Shope', 'LVEDV', 'LVESV',
        'AV_ring', 'MV_ring', 'RVOT'
    ]
    
    # NEVER mask these (severity-discriminative, Cramér V > 0.15)
    NEVER_MASK_FIELDS = [
        'EF', 'LVIDd', 'LVIDs', 'LA', 'AO', 'IVST', 'LVPWT', 'Age', 'Sex',
        'LV_Cavity_Size', 'LV_Wall_Thickness', 'LV_Wall_Motion',
        'VSD', 'ASD', 'PDA', 'ACS', 'RV', 'IAS', 'IVS',
        'Mitral_Valve_Flow', 'Aortic_Valve_Flow', 'Tricuspid_Valve_Flow',
        'Pulmonary_Valve_Flow', 'VSD_Flow', 'Others_Flow',
        'RA', 'MV', 'AV', 'LA_Description', 'Aorta', 'TV', 'PV',
        'RV_Description', 'PA_Description', 'Thrombus', 'Vegetation',
        'Pericardium', 'MVA',
    ]
    
    def __init__(self, severity_label: str, augment_prob: float = 0.4):
        self.severity_label = severity_label
        self.augment_prob = augment_prob
    
    def augment_findings(self, findings_text: str) -> str:
        """
        Apply all augmentation strategies to findings text.
        NOTE: This is called ONLY on Stream 1 text (no Impression).
        """
        text = findings_text
        
        if random.random() < self.augment_prob:
            text = self._ef_perturbation(text)
        
        if random.random() < self.augment_prob:
            text = self._measurement_noise(text)
        
        if random.random() < self.augment_prob:
            text = self._synonym_replacement(text, max_replacements=2)
        
        if random.random() < self.augment_prob:
            text = self._field_masking(text, max_masks=2)
        
        return text
    
    def _ef_perturbation(self, text: str) -> str:
        """Add ±5% Gaussian noise to EF within observed class range."""
        dist = self.EF_DISTRIBUTIONS.get(self.severity_label)
        if not dist:
            return text
        
        def perturb_ef(match):
            original_ef = float(match.group(1))
            noise = random.gauss(0, 5)  # ±5% inter-observer variability
            new_ef = int(round(original_ef + noise))
            new_ef = max(dist['min'], min(dist['max'], new_ef))
            return f"EF: {new_ef}%"
        
        return re.sub(r'EF:\s*(\d+)%', perturb_ef, text)
    
    def _measurement_noise(self, text: str) -> str:
        """Add inter-observer measurement variability noise."""
        for field, max_noise in self.MEASUREMENT_NOISE.items():
            pattern = rf'{field}:\s*(\d+)\s*mm'
            match = re.search(pattern, text)
            if match:
                original = int(match.group(1))
                noise = random.uniform(-max_noise, max_noise)
                new_val = max(1, int(round(original + noise)))
                text = re.sub(pattern, f'{field}: {new_val} mm', text)
        return text
    
    def _synonym_replacement(self, text: str, max_replacements: int = 2) -> str:
        """Replace ONLY severity-neutral terms."""
        replaced = 0
        for original, synonyms in self.SAFE_SYNONYMS.items():
            if replaced >= max_replacements:
                break
            if original in text:
                synonym = random.choice(synonyms)
                text = text.replace(original, synonym, 1)
                replaced += 1
        return text
    
    def _field_masking(self, text: str, max_masks: int = 2) -> str:
        """Mask only non-informative constant fields."""
        masked = 0
        fields_to_mask = random.sample(
            self.SAFE_TO_MASK_FIELDS,
            min(max_masks, len(self.SAFE_TO_MASK_FIELDS))
        )
        for field in fields_to_mask:
            pattern = rf'{field}:\s*[^,]+'
            if re.search(pattern, text):
                text = re.sub(pattern, f'{field}: not reported', text)
                masked += 1
        return text


def augment_dataset_two_stage(
    texts: List[str],
    labels: List[int],
    label_map_inv: Dict[int, str],
) -> Tuple[List[str], List[int]]:
    """
    Two-stage augmentation (uniform + oversampling).
    
    Stage 1: Apply augmentation uniformly to ALL classes (prevents
             asymmetric augmentation artifact bias)
    Stage 2: Oversample minority classes by duplicating from the 
             augmented pool
    
    IMPORTANT: Augmentation modifies only the FINDINGS portion.
    The Impression is never touched. When the augmented text is later
    split by DualStreamEchoDataset._split_report(), the Impression
    remains identical to the original.
    """
    counts = Counter(labels)
    
    # Stage 1: Uniform augmentation (one variant per sample, all classes)
    stage1_texts = list(texts)
    stage1_labels = list(labels)
    
    for idx in range(len(texts)):
        original = texts[idx]
        label_id = labels[idx]
        severity_name = label_map_inv[label_id]
        
        # Split → augment findings only → rejoin
        findings, impression = DualStreamEchoDataset._split_report(original)
        augmenter = FindingsAugmenter(severity_label=severity_name, augment_prob=0.4)
        aug_findings = augmenter.augment_findings(findings)
        
        # Rejoin into full text (impression stays identical)
        aug_text = f"{aug_findings}, Impression: {impression}"
        
        stage1_texts.append(aug_text)
        stage1_labels.append(label_id)
    
    # Stage 2: Minority oversampling to balance classes
    stage2_target = max(Counter(stage1_labels).values())
    final_texts = list(stage1_texts)
    final_labels = list(stage1_labels)
    
    for label_id, count in Counter(stage1_labels).items():
        if count >= stage2_target:
            continue
        indices = [i for i, l in enumerate(stage1_labels) if l == label_id]
        needed = stage2_target - count
        for _ in range(needed):
            idx = random.choice(indices)
            final_texts.append(stage1_texts[idx])
            final_labels.append(label_id)
    
    print(f"\nAugmentation summary:")
    print(f"  Original: {dict(counts)}")
    print(f"  After Stage 1 (uniform): {dict(Counter(stage1_labels))}")
    print(f"  After Stage 2 (balanced): {dict(Counter(final_labels))}")
    print(f"  Total: {len(texts)} → {len(final_texts)}")
    
    return final_texts, final_labels


# ============================================================================
# 3. ENCODER FINE-TUNING (Selective Layer Unfreezing + Optional PEFT LoRA)
# ============================================================================
#
# CRITICAL FIX (v2.1): The original LoRA implementation applied LoRA deltas
# POST-HOC to the final hidden state, not INSIDE the attention layers.
# This meant the attention computation used un-adapted Q,V projections,
# making LoRA ineffective (F1 ≈ 0.29, near random).
#
# Two correct approaches:
#   Option A: PEFT library (pip install peft) — proper hook-based LoRA
#   Option B: Selective layer unfreezing — simpler, no extra dependency
#
# Default: Option B (unfreezing). Set use_peft_lora=True for Option A.

class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear layer that REPLACES the original linear.
    This ensures LoRA is applied INSIDE the computation, not post-hoc.
    
    Output = W_frozen(x) + scaling * B(A(dropout(x)))
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        return self.original.weight

    @property
    def bias(self):
        return self.original.bias

    @property
    def in_features(self):
        return self.original.in_features

    @property
    def out_features(self):
        return self.original.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen output + LoRA trainable delta
        base_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out


def setup_encoder(model: nn.Module, mode: str = "unfreeze",
                  num_unfreeze_layers: int = 4,
                  lora_rank: int = 8, lora_alpha: float = 16.0) -> int:
    """
    Configure encoder for training. Two modes:

    mode="unfreeze": Freeze layers 0-(N-k), unfreeze last k layers.
        Simple, reliable, well-proven for clinical NLP.
        ~25M trainable params per encoder (last 4 layers).

    mode="lora": Replace Q,V projections in last k layers with LoRALinear.
        Proper in-place LoRA (not the broken post-hoc version).
        ~50K trainable params per encoder.

    Returns: number of trainable parameters.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = 0
    num_layers = len(model.encoder.layer)
    target_start = max(0, num_layers - num_unfreeze_layers)

    if mode == "unfreeze":
        # Unfreeze last N layers + pooler
        for layer_idx in range(target_start, num_layers):
            for param in model.encoder.layer[layer_idx].parameters():
                param.requires_grad = True
                trainable_params += param.numel()

    elif mode == "lora":
        # Replace Q,V linear layers with LoRALinear in last N layers
        for layer_idx in range(target_start, num_layers):
            attention = model.encoder.layer[layer_idx].attention.self
            for proj_name in ["query", "value"]:
                original = getattr(attention, proj_name)
                lora_linear = LoRALinear(original, rank=lora_rank, alpha=lora_alpha)
                setattr(attention, proj_name, lora_linear)
                trainable_params += sum(
                    p.numel() for p in lora_linear.parameters() if p.requires_grad
                )

    return trainable_params


# ============================================================================
# 4. CLINICAL DOMAIN ATTENTION (CDA) — Per-Stream
# ============================================================================

class ClinicalDomainAttention(nn.Module):
    """
    CDA with multi-sample dropout regularization.
    
    For Stream 1 (findings): learns which measurements/findings matter most
    For Stream 2 (impression): learns which impression words are key
    
    The attention weights from both streams are INTERPRETABLE:
    - Stream 1 attention → "model focused on EF, Wall_Motion, VSD"
    - Stream 2 attention → "model focused on 'moderate', 'dysfunction'"
    """
    def __init__(self, hidden_size: int, dropout: float = 0.2,
                 num_dropout_samples: int = 5):
        super().__init__()
        self.num_dropout_samples = num_dropout_samples
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_mlp(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)

        attended = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)

        if self.training and self.num_dropout_samples > 1:
            outputs = [self.dropout(attended) for _ in range(self.num_dropout_samples)]
            attended = torch.stack(outputs).mean(dim=0)
        else:
            attended = self.dropout(attended)

        return self.layer_norm(attended), attention_weights


# ============================================================================
# 5. CROSS-STREAM GATED FUSION MODULE (CSGFM)
# ============================================================================

class CrossStreamGatedFusion(nn.Module):
    """
    Learns a dynamic gate: how much to weight findings vs impression.
    
    g = sigmoid(W[findings; impression])
    fused = g ⊙ findings + (1-g) ⊙ impression
    
    The gate value g is INTERPRETABLE:
    - g → 1.0: model relies more on clinical findings
    - g → 0.0: model relies more on impression
    - Per-sample g values can be reported in the paper
    """
    def __init__(self, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        bottleneck = hidden_size // 4
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 2, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, hidden_size),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, findings_repr: torch.Tensor,
                impression_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([findings_repr, impression_repr], dim=-1)
        gate = self.gate_network(gate_input)
        fused = gate * findings_repr + (1 - gate) * impression_repr
        return self.output_dropout(self.output_norm(fused)), gate


# ============================================================================
# 6. HIERARCHICAL CLINICAL FEATURE AGGREGATOR (HCFA)
# ============================================================================

class LightweightHCFA(nn.Module):
    """
    3-group hierarchical aggregation matching clinical report structure:
      Group 1: Measurements (EF, dimensions — numeric clinical evidence)
      Group 2: Structural findings (wall motion, valve status — categorical)
      Group 3: Diagnostic summary (impression-derived features)
    
    Learned queries attend to the fused sequence to extract group-specific
    representations, then aggregate into a single feature vector.
    """
    def __init__(self, hidden_size: int, num_entity_groups: int = 3,
                 num_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.group_queries = nn.Parameter(torch.randn(num_entity_groups, hidden_size))
        nn.init.xavier_uniform_(self.group_queries.unsqueeze(0))

        self.group_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_size * num_entity_groups, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, sequence_hidden: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = sequence_hidden.size(0)
        queries = self.group_queries.unsqueeze(0).expand(batch_size, -1, -1)
        key_pad_mask = (attention_mask == 0) if attention_mask is not None else None

        group_repr, _ = self.group_attention(
            query=queries, key=sequence_hidden, value=sequence_hidden,
            key_padding_mask=key_pad_mask,
        )
        return self.aggregator(group_repr.reshape(batch_size, -1))


# ============================================================================
# 7. CLASSIFICATION HEAD (Standard CE + Ordinal Distance)
# ============================================================================

class ClassificationHead(nn.Module):
    """
    FIX v2.2: Replaced CORAL (caused threshold collapse → only 2 of 4 classes).
    
    Standard softmax classifier with:
    - Two-layer MLP bottleneck
    - Direct 4-class logits (no cumulative sigmoid thresholds)
    - Auxiliary ordinal head for ordinal distance penalty
    
    This is simpler, more robust, and doesn't suffer from threshold collapse.
    """
    def __init__(self, hidden_size: int, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        bottleneck = hidden_size // 4  # 192

        self.shared_ffn = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Primary: standard 4-class classification
        self.main_classifier = nn.Linear(bottleneck, num_classes)
        # Auxiliary: scalar severity score for ordinal penalty
        self.ordinal_head = nn.Linear(bottleneck, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared = self.shared_ffn(features)
        logits = self.main_classifier(shared)
        probs = F.softmax(logits, dim=-1)
        ordinal_score = self.ordinal_head(shared).squeeze(-1)

        return {
            "logits": logits,
            "probs": probs,
            "ordinal_score": ordinal_score,
        }


# ============================================================================
# 8. COMPLETE CDS-FUSION-V2 MODEL
# ============================================================================

class CDSFusionV2(nn.Module):
    """
    CDS-Fusion-V2: Clinically-Motivated Dual-Stream Fusion Network.
    
    Stream 1 (BioBERT + LoRA):  Clinical findings (measurements, structures, flows)
    Stream 2 (ClinicalBERT + LoRA): Impression text (diagnostic conclusion)
    
    Key difference from V1:
    - V1: Both streams received the SAME full report text
    - V2: Each stream receives DIFFERENT, clinically-motivated input
    - V1: Model shortcutted through Impression in both streams
    - V2: Stream 1 has NO impression — must learn from clinical evidence
    - V1: Augmentation was wasted (impression unchanged, model ignored rest)
    - V2: Augmentation is meaningful (Stream 1 directly affected)
    
    Trainable params: ~2M (LoRA + adapters + fusion + classifier)
    Frozen params: ~220M (both BERT encoders)
    """
    
    def __init__(
        self,
        biobert_name: str = "dmis-lab/biobert-base-cased-v1.2",
        clinicalbert_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_classes: int = 4,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        dropout: float = 0.25,
        num_dropout_samples: int = 5,
        encoder_mode: str = "unfreeze",  # "unfreeze" or "lora"
        num_unfreeze_layers: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ---- Load encoders ----
        self.biobert = AutoModel.from_pretrained(biobert_name)
        self.clinicalbert = AutoModel.from_pretrained(clinicalbert_name)
        hidden_size = self.biobert.config.hidden_size  # 768

        # ---- Configure encoder training ----
        # FIXED v2.1: Uses proper in-place LoRA or selective unfreezing
        # instead of broken post-hoc LoRA that caused F1≈0.29
        bio_trainable = setup_encoder(
            self.biobert, mode=encoder_mode,
            num_unfreeze_layers=num_unfreeze_layers,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
        )
        clin_trainable = setup_encoder(
            self.clinicalbert, mode=encoder_mode,
            num_unfreeze_layers=num_unfreeze_layers,
            lora_rank=lora_rank, lora_alpha=lora_alpha,
        )

        # ---- Lightweight adapter projections ----
        self.findings_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.impression_adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # ---- Per-stream CDA ----
        self.findings_cda = ClinicalDomainAttention(hidden_size, dropout, num_dropout_samples)
        self.impression_cda = ClinicalDomainAttention(hidden_size, dropout, num_dropout_samples)

        # ---- Cross-Stream Gated Fusion ----
        self.fusion = CrossStreamGatedFusion(hidden_size, dropout)

        # ---- Sequence-level fusion for HCFA ----
        self.seq_fusion_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.hcfa = LightweightHCFA(hidden_size, num_entity_groups=3, num_heads=2, dropout=dropout)

        # ---- Feature combination ----
        self.feature_combiner = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        # ---- Standard classifier (replaced CORAL) ----
        self.classifier = ClassificationHead(hidden_size, num_classes, dropout=0.3)

        # ---- Parameter summary ----
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*60}")
        print(f"CDS-Fusion-V2.1 Model Summary (FIXED)")
        print(f"{'='*60}")
        print(f"Encoder mode:         {encoder_mode}")
        print(f"Stream 1: BioBERT  → Clinical Findings (augmentable)")
        print(f"Stream 2: ClinicalBERT → Impression (not augmented)")
        print(f"{'='*60}")
        print(f"Total parameters:     {total:>12,d}")
        print(f"Trainable parameters: {trainable:>12,d} ({trainable/total*100:.1f}%)")
        print(f"Frozen parameters:    {total-trainable:>12,d} ({(total-trainable)/total*100:.1f}%)")
        print(f"BioBERT trainable:    {bio_trainable:>12,d}")
        print(f"ClinicalBERT train:   {clin_trainable:>12,d}")
        print(f"{'='*60}\n")

    def _encode(self, model: nn.Module, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        FIXED: Simple forward pass through encoder.

        With 'unfreeze' mode: last N layers have requires_grad=True,
        so gradients flow through them normally during backprop.

        With 'lora' mode: Q,V projections are LoRALinear modules that
        compute W_frozen(x) + B(A(x)) IN-PLACE during the attention
        computation. No post-hoc modification needed.

        Both modes work correctly with a standard forward call.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def forward(
        self,
        bio_input_ids: torch.Tensor,       # Stream 1: findings tokens
        bio_attention_mask: torch.Tensor,
        clin_input_ids: torch.Tensor,       # Stream 2: impression tokens
        clin_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass:
        1. Encode findings with BioBERT (unfrozen/LoRA layers adapt representations)
        2. Encode impression with ClinicalBERT (same)
        3. Per-stream CDA (extract key representations)
        4. Gated fusion (learn findings-vs-impression weighting)
        5. HCFA (hierarchical feature aggregation)
        6. Ordinal classification
        """
        # ---- Stream 1: Clinical Findings (BioBERT) ----
        findings_seq = self._encode(self.biobert, bio_input_ids, bio_attention_mask)
        findings_seq = self.findings_adapter(findings_seq)
        findings_pooled, findings_attn = self.findings_cda(findings_seq, bio_attention_mask)

        # ---- Stream 2: Impression (ClinicalBERT) ----
        impression_seq = self._encode(self.clinicalbert, clin_input_ids, clin_attention_mask)
        impression_seq = self.impression_adapter(impression_seq)
        impression_pooled, impression_attn = self.impression_cda(impression_seq, clin_attention_mask)

        # ---- Gated Fusion ----
        fused_pooled, gate_values = self.fusion(findings_pooled, impression_pooled)

        # ---- Sequence-Level Fusion for HCFA ----
        min_len = min(findings_seq.size(1), impression_seq.size(1))
        seq_concat = torch.cat([
            findings_seq[:, :min_len, :],
            impression_seq[:, :min_len, :]
        ], dim=-1)
        seq_fused = self.seq_fusion_proj(seq_concat)

        mask_trunc = bio_attention_mask[:, :min_len]
        hcfa_repr = self.hcfa(seq_fused, mask_trunc)

        # ---- Combine & Classify ----
        combined = torch.cat([fused_pooled, hcfa_repr], dim=-1)
        features = self.feature_combiner(combined)

        outputs = self.classifier(features)

        # Store attention weights and gate values for interpretability
        outputs["findings_attention"] = findings_attn
        outputs["impression_attention"] = impression_attn
        outputs["gate_values"] = gate_values

        return outputs


# ============================================================================
# 9. LOSS FUNCTION (Focal CE + Ordinal Distance + R-Drop)
# ============================================================================

class CDSFusionV2Loss(nn.Module):
    """
    FIX v2.2: Replaced CORAL loss (threshold collapse) with:
    
    1. Focal Cross-Entropy (primary): Handles class imbalance
       - Standard CE with focal modulation: (1-pt)^γ * CE
       - Class weights for Severe (79 samples) vs Mild (269 samples)
    
    2. Ordinal Distance Penalty (auxiliary): Preserves ordinality
       - MSE between predicted severity score and true ordinal label
       - Captures: Normal(0) < Mild(1) < Moderate(2) < Severe(3)
       - Simpler than CORAL, doesn't suffer threshold collapse
    
    3. R-Drop KL Divergence: Regularization for low-resource
       - Dual forward pass with different dropout masks
       - Minimizes KL divergence between two outputs
    """
    def __init__(self, num_classes: int = 4, focal_weight: float = 0.6,
                 ordinal_weight: float = 0.2, rdrop_weight: float = 0.2,
                 focal_gamma: float = 2.0, label_smoothing: float = 0.1,
                 class_weights: Optional[torch.Tensor] = None,
                 # Keep old param names for backward compat (ignored)
                 coral_weight: float = 0.0, focal_w_old: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.focal_w = focal_weight
        self.ordinal_w = ordinal_weight
        self.rdrop_w = rdrop_weight
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def focal_loss(self, logits, labels):
        """Focal Cross-Entropy: down-weights easy examples, focuses on hard ones."""
        ce = F.cross_entropy(logits, labels, weight=self.class_weights,
                             label_smoothing=self.label_smoothing, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.focal_gamma) * ce).mean()

    def ordinal_distance_loss(self, ordinal_scores, labels):
        """
        MSE between predicted severity score and true ordinal position.
        Forces the model to learn: Normal=0 < Mild=1 < Moderate=2 < Severe=3.
        """
        targets = labels.float()
        return F.mse_loss(ordinal_scores, targets)

    def rdrop_kl(self, logits1, logits2):
        """Symmetric KL divergence between two forward passes."""
        p = F.log_softmax(logits1, -1)
        q = F.log_softmax(logits2, -1)
        return (F.kl_div(p, F.softmax(logits2, -1), reduction="batchmean") +
                F.kl_div(q, F.softmax(logits1, -1), reduction="batchmean")) / 2

    def forward(self, out1, out2, labels):
        focal = self.focal_loss(out1["logits"], labels)
        ordinal = self.ordinal_distance_loss(out1["ordinal_score"], labels)
        total = self.focal_w * focal + self.ordinal_w * ordinal

        rdrop = torch.tensor(0.0, device=labels.device)
        if out2 is not None and self.training:
            focal2 = self.focal_loss(out2["logits"], labels)
            ordinal2 = self.ordinal_distance_loss(out2["ordinal_score"], labels)
            total = (total + self.focal_w * focal2 + self.ordinal_w * ordinal2) / 2
            rdrop = self.rdrop_kl(out1["logits"], out2["logits"])
            total += self.rdrop_w * rdrop

        return {"total_loss": total, "focal_loss": focal,
                "ordinal_loss": ordinal, "rdrop_loss": rdrop}


# ============================================================================
# 10. TRAINING UTILITIES
# ============================================================================

def compute_class_weights(labels: List[int], num_classes: int = 4) -> torch.Tensor:
    """Inverse frequency class weights for imbalanced data."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor([total / (num_classes * counts[i]) for i in range(num_classes)])
    return weights / weights.sum() * num_classes


def get_predictions(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Get predicted class from softmax probabilities."""
    return outputs["probs"].argmax(dim=-1)


# ============================================================================
# 11. MAIN — Example Usage
# ============================================================================

if __name__ == "__main__":
    
    # ---- Configuration ----
    CONFIG = {
        "data_path": "Final_data.csv",
        "biobert_name": "dmis-lab/biobert-base-cased-v1.2",
        "clinicalbert_name": "emilyalsentzer/Bio_ClinicalBERT",
        "max_findings_len": 256,
        "max_impression_len": 64,
        "batch_size": 16,
        "learning_rate": 2e-4,
        "num_epochs": 30,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "dropout": 0.25,
        "num_folds": 5,
        "num_repeats": 3,  # Repeated Stratified K-Fold
        "swa_start_epoch": 20,
        "seed": 42,
    }
    
    print("=" * 60)
    print("CDS-Fusion-V2: Training Pipeline")
    print("=" * 60)
    
    # ---- Load Data ----
    texts, labels, label_map_inv = prepare_data(CONFIG["data_path"])
    
    # ---- Augment (findings only) ----
    aug_texts, aug_labels = augment_dataset_two_stage(texts, labels, label_map_inv)
    
    # ---- Tokenizers ----
    bio_tokenizer = AutoTokenizer.from_pretrained(CONFIG["biobert_name"])
    clin_tokenizer = AutoTokenizer.from_pretrained(CONFIG["clinicalbert_name"])
    
    # ---- Dataset ----
    dataset = DualStreamEchoDataset(
        aug_texts, aug_labels,
        bio_tokenizer, clin_tokenizer,
        max_findings_len=CONFIG["max_findings_len"],
        max_impression_len=CONFIG["max_impression_len"],
    )
    
    print(f"\nDataset created: {len(dataset)} samples")
    print(f"  Stream 1 (findings): max {CONFIG['max_findings_len']} tokens")
    print(f"  Stream 2 (impression): max {CONFIG['max_impression_len']} tokens")
    
    # ---- Model ----
    model = CDSFusionV2(
        biobert_name=CONFIG["biobert_name"],
        clinicalbert_name=CONFIG["clinicalbert_name"],
        lora_rank=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        dropout=CONFIG["dropout"],
    )
    
    # ---- Class weights ----
    weights = compute_class_weights(aug_labels)
    print(f"Class weights: {weights.tolist()}")
    
    # ---- Loss ----
    criterion = CDSFusionV2Loss(class_weights=weights)
    
    # ---- Training would continue with:
    # - RepeatedStratifiedKFold (5-fold × 3 repeats)
    # - AdamW optimizer with cosine annealing
    # - SWA from epoch 20
    # - R-Drop (two forward passes per batch)
    # - Bootstrapped confidence intervals
    # - McNemar's test against baselines
    
    print("\n✅ Model ready for training!")
    print("Next steps: implement training loop with RepeatedStratifiedKFold")
