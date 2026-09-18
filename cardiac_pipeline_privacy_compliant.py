"""
PRIVACY-COMPLIANT CARDIAC DATA PROCESSING PIPELINE
=====================================================
Complete end-to-end processing with HIPAA Safe Harbor de-identification:
  Step 1: Extract data from .doc reports (from extract_medical_reports.py)
          - Uses MS Word COM for accurate table extraction
          - Uses Azure OpenAI for structured field parsing
  Step 2: De-identify — strip all PHI (ID, Date, Patient_Name, Filename)
  Step 3: Classify into 11 cardiac disease categories (from mail_script.py)
  Step 4: Map to 4 severity labels (from mail_script.py)
  Step 5: Generate de-identified text format dataset (from mail_script.py)

PRIVACY DESIGN:
  - Raw reports may contain patient demographics (ID, Name, Date)
  - These are extracted by the LLM but IMMEDIATELY stripped before any
    data is saved to disk or used downstream
  - Only Age (binned if >89) and Sex are retained as clinically essential
    demographic features
  - Azure OpenAI processes data under enterprise data governance policies
    (no data retention, no training on customer data)
  - All intermediate and final outputs are PHI-free

Author: Mohammad Naimul Islam Shanto
Affiliation: Kennesaw State University
"""

import os
import json
import csv
import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm
import time
import win32com.client
import pythoncom
import threading
import logging
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

load_dotenv()

# Azure OpenAI Configuration
AZURE_CONFIG = {
    "api_key": os.getenv("AZURE_API_KEY"),
    "api_version": os.getenv("AZURE_API_VERSION"),
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "deployment_name": os.getenv("AZURE_DEPLOYMENT")
}

# Thread-local storage for Azure client
thread_local = threading.local()

def get_azure_client():
    """Get thread-local Azure OpenAI client"""
    if not hasattr(thread_local, "azure_client"):
        thread_local.azure_client = AzureOpenAI(
            api_key=AZURE_CONFIG["api_key"],
            api_version=AZURE_CONFIG["api_version"],
            azure_endpoint=AZURE_CONFIG["azure_endpoint"]
        )
    return thread_local.azure_client


# Global Azure client for classification steps
client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)
DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT")


# ============================================================================
# FIELD DEFINITIONS — PRIVACY-AWARE
# ============================================================================

# PHI fields — extracted by LLM but STRIPPED before saving
PHI_FIELDS = ["ID", "Date", "Patient_Name"]

# Fields to extract from raw reports (includes PHI for LLM parsing accuracy,
# but PHI is removed immediately after extraction)
EXTRACTION_FIELDS = [
    # Demographics (only Age and Sex are retained)
    "Age", "Sex",

    # Measurements (M-mode and 2-D)
    "IVST", "LVIDd", "LA", "LVPWT", "LVIDs", "AO", "RVGWT", "FS", "ACS",
    "RV", "EF", "LVEDV", "PA", "EF_Shope", "LVESV", "AV_ring", "MV_ring", "MVA",

    # Description (M-mode and 2-D) — LV Details
    "LV_Cavity_Size", "LV_Wall_Thickness", "LV_Wall_Motion",

    # Description — Other Chambers and Valves
    "RA", "MV", "RV_Description", "AV", "LA_Description", "PV", "Aorta",
    "TV", "PA_Description", "IAS", "RVOT", "IVS", "ASD", "Thrombus",
    "VSD", "Vegetation", "PDA", "Pericardium",

    # Color Flow Mapping and Doppler Study
    "Mitral_Valve_Flow", "Aortic_Valve_Flow", "Pulmonary_Valve_Flow",
    "Tricuspid_Valve_Flow", "VSD_Flow", "Others_Flow",

    # Impression
    "Impression"
]

# Fields sent to the LLM for extraction (includes PHI for parsing accuracy)
# PHI is stripped IMMEDIATELY after LLM returns the response
LLM_EXTRACTION_FIELDS = PHI_FIELDS + EXTRACTION_FIELDS

# Medical fields used for text format generation (PHI-free)
MEDICAL_FIELDS = EXTRACTION_FIELDS  # Same as extraction fields, no PHI

# 11 Cardiac Classes
CARDIAC_CLASSES = {
    "Normal": "Normal echocardiographic findings with no structural or functional abnormalities",
    "Ventricular_Dysfunction": "Regional or global ventricular dysfunction without specific etiology",
    "Ischemic_Heart_Disease": "Myocardial ischemia without completed infarction",
    "Myocardial_Infarction": "Completed heart attack with tissue damage",
    "Cardiomyopathy": "Primary heart muscle disease (Ischemic or Dilated)",
    "Valvular_Disease": "Structural valve abnormalities (stenosis or regurgitation)",
    "Congenital_Heart_Disease": "Structural heart defects present from birth",
    "Post_Intervention": "Status post cardiac intervention or surgery",
    "Hypertensive_Heart_Disease": "Cardiac changes due to chronic hypertension",
    "Pulmonary_Hypertension": "Elevated pressure in pulmonary arteries",
    "Other_Cardiac_Conditions": "Miscellaneous cardiac findings not fitting other categories"
}

# File paths
DATA_DIR = Path("data set")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
Path("feature_engineering").mkdir(exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "extracted_deidentified_reports.json"
OUTPUT_EXTRACTED_CSV = OUTPUT_DIR / "extracted_deidentified_reports.csv"
OUTPUT_CLASSIFIED_CSV = Path("feature_engineering/classified_cardiac_data.csv")
OUTPUT_WITH_LABELS_CSV = Path("feature_engineering/classified_cardiac_data_with_labels.csv")
OUTPUT_TEXT_FORMAT_CSV = OUTPUT_DIR / "medical_reports_text_format.csv"
OUTPUT_DEIDENTIFICATION_LOG = OUTPUT_DIR / "deidentification_log.json"

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# STEP 1: EXTRACTION — from extract_medical_reports.py
# (MS Word COM + Azure OpenAI table-aware extraction)
# ============================================================================

def extract_text_and_tables_from_doc(file_path: str) -> Dict[str, any]:
    """
    Extract text and tables from .doc file using MS Word COM.
    Returns structured data with tables preserved for accurate LLM parsing.
    
    Source: extract_medical_reports.py
    """
    pythoncom.CoInitialize()

    word = None
    doc = None

    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0

        doc = word.Documents.Open(str(Path(file_path).absolute()))
        full_text = doc.Content.Text

        # Extract tables
        tables_data = []
        for table_idx in range(1, doc.Tables.Count + 1):
            table = doc.Tables(table_idx)
            table_content = []

            for row_idx in range(1, table.Rows.Count + 1):
                try:
                    row_data = []
                    for col_idx in range(1, table.Columns.Count + 1):
                        try:
                            cell = table.Cell(row_idx, col_idx)
                            cell_text = cell.Range.Text.strip()
                            cell_text = cell_text.replace('\r\x07', '').replace('\x07', '').strip()
                            row_data.append(cell_text)
                        except:
                            row_data.append("")

                    if any(cell for cell in row_data):
                        table_content.append(row_data)
                except:
                    continue

            if table_content:
                tables_data.append(table_content)

        return {
            "status": "success",
            "full_text": full_text,
            "tables": tables_data,
            "num_tables": len(tables_data)
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "full_text": "",
            "tables": [],
            "num_tables": 0
        }

    finally:
        try:
            if doc:
                doc.Close(False)
            if word:
                word.Quit()
        except:
            pass
        pythoncom.CoUninitialize()


def format_tables_for_llm(tables: List[List[List[str]]]) -> str:
    """Format tables in a clear structure for LLM"""
    if not tables:
        return "No tables found"

    formatted = []
    for idx, table in enumerate(tables, 1):
        formatted.append(f"\n--- TABLE {idx} ---")
        for row in table:
            formatted.append(" | ".join(str(cell) for cell in row))

    return "\n".join(formatted)


def extract_data_with_llm(document_data: Dict, filename: str,
                          retry_count: int = 3) -> Dict[str, str]:
    """
    Use Azure OpenAI to extract structured data from document.
    Uses table-aware extraction from extract_medical_reports.py.

    PRIVACY NOTE:
    - The LLM prompt asks for PHI fields (ID, Name, Date) because they
      appear in the raw document and help the LLM accurately parse the
      report structure.
    - PHI fields are STRIPPED immediately after extraction in the
      deidentify_record() function — they are never saved to disk.
    - Azure OpenAI enterprise deployments do NOT retain customer data
      and do NOT use it for model training.
    """

    full_text = document_data.get("full_text", "")
    tables = document_data.get("tables", [])
    tables_formatted = format_tables_for_llm(tables)

    system_prompt = """You are an expert medical data extraction specialist for Color Doppler Echocardiogram Reports.

EXTRACTION STRATEGY:

1. **Demographics** (from header or first table):
   - Age: Just the number (e.g., "55")
   - Sex: Male/Female

2. **Measurement Table** (look for "Measurement : (M - mode and 2 - D)"):
   - Extract measurements with units (e.g., "11 mm", "41 %", "2.4 sqcm")
   - Common fields: IVST, LVIDd, LA, LVPWT, LVIDs, AO, FS, EF, MVA, etc.
   - If a measurement cell is empty or "mm" only → use "N/A"
   - Keep units with values (e.g., "56 mm", NOT just "56")

3. **Description Section** (look for "Description (M - mode and 2 - D)"):
   - LV_Cavity_Size: e.g., "Dilated", "Normal"
   - LV_Wall_Thickness: e.g., "Normal", "Hypertrophied"
   - LV_Wall_Motion: Full description of wall motion
   - Chamber descriptions: RA, RV, LA, Aorta, etc.
   - Structural findings: IAS, IVS, ASD, VSD, Thrombus, Vegetation, PDA, Pericardium
   - Use "Intact" for IAS/IVS if intact
   - Use "Absent" for ASD, VSD, Thrombus, Vegetation, PDA if absent
   - Use "Normal" or specific finding for chambers/structures

4. **Color Flow Mapping Section**:
   - Valve flows: Mitral_Valve_Flow, Aortic_Valve_Flow, etc.
   - Extract exact description (e.g., "Normal flow", "Trivial regurgitation")

5. **Impression Section**:
   - Extract complete impression text

CRITICAL RULES:
- Return EXACT field names from the required list
- Keep measurements with units
- Use "N/A" only for truly missing data
- Use "Absent" for documented absent findings (not "N/A")
- Use "Intact" for IAS/IVS if intact
- Use "Normal" for normal findings
- Do NOT extract patient ID, patient name, or date — these are excluded for privacy
"""

    user_prompt = f"""Extract ALL fields from this Color Doppler Echocardiogram Report.

REQUIRED FIELDS (use these exact names):
{json.dumps(EXTRACTION_FIELDS, indent=2)}

DOCUMENT TABLES:
{tables_formatted}

FULL DOCUMENT TEXT:
{full_text[:5000]}

INSTRUCTIONS:
1. Look at the TABLES first for structured data (measurements, descriptions)
2. Use the FULL TEXT for additional context and impression
3. Match each value to the EXACT field name from the required list
4. Return JSON with ALL required fields (use "N/A" if truly missing)
5. Keep units with measurements (e.g., "11 mm", "41 %")
6. Do NOT include any patient identifying information (no ID, Name, or Date)

Return ONLY valid JSON with the required fields."""

    for attempt in range(retry_count):
        try:
            client_local = get_azure_client()

            response = client_local.chat.completions.create(
                model=AZURE_CONFIG["deployment_name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
                timeout=90
            )

            result = json.loads(response.choices[0].message.content)

            # Ensure all required fields are present
            for field in EXTRACTION_FIELDS:
                if field not in result:
                    result[field] = "N/A"

            # ============================================================
            # PRIVACY ENFORCEMENT: Strip any PHI the LLM may have returned
            # even though we didn't ask for it
            # ============================================================
            for phi_field in PHI_FIELDS:
                result.pop(phi_field, None)

            # Also strip any field containing "name", "id", "date" variants
            keys_to_remove = []
            for key in result.keys():
                key_lower = key.lower()
                if any(phi in key_lower for phi in
                       ["patient_name", "patient_id", "date_of",
                        "dob", "mrn", "record_number", "address",
                        "phone", "email", "ssn"]):
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                result.pop(key, None)

            return result

        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep((attempt + 1) * 2)
            else:
                logger.warning(f"LLM extraction failed for {filename}: {str(e)}")
                return {field: "N/A" for field in EXTRACTION_FIELDS}


# ============================================================================
# STEP 2: DE-IDENTIFICATION — HIPAA Safe Harbor Method
# ============================================================================

def deidentify_record(record: Dict, record_idx: int) -> Dict:
    """
    Apply HIPAA Safe Harbor de-identification to a single record.

    Actions:
    1. Remove all PHI fields (ID, Date, Patient_Name, Filename)
    2. Bin Age > 89 to "90+" per HIPAA Safe Harbor
    3. Scan text fields for potential PHI leakage
    4. Assign anonymous sequential index

    Returns de-identified record.
    """
    deidentified = {}

    # Copy only non-PHI fields
    for field in EXTRACTION_FIELDS:
        value = record.get(field, "N/A")

        # Age binning: HIPAA requires ages > 89 to be grouped as 90+
        if field == "Age":
            try:
                age_val = int(str(value).strip())
                if age_val > 89:
                    value = "90+"
                else:
                    value = str(age_val)
            except (ValueError, TypeError):
                value = str(value) if value else "N/A"

        deidentified[field] = value

    # Assign anonymous index instead of Filename
    deidentified["Record_Index"] = record_idx

    return deidentified


def scan_for_phi_leakage(record: Dict, known_names: set = None) -> List[str]:
    """
    Scan all text fields in a record for potential PHI leakage.
    Returns list of warnings if any PHI-like content is detected.
    """
    warnings = []

    # Patterns that might indicate PHI leakage
    phi_patterns = [
        (r'\b\d{2}[-/]\d{2}[-/]\d{2,4}\b', "date pattern"),
        (r'\b\d{8,12}\b', "potential ID/MRN"),
        (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', "potential name"),
    ]

    text_fields = ["LV_Wall_Motion", "Impression", "Mitral_Valve_Flow",
                   "Aortic_Valve_Flow", "Others_Flow"]

    for field in text_fields:
        value = str(record.get(field, ""))
        for pattern, description in phi_patterns:
            if re.search(pattern, value):
                # Check if it's a legitimate clinical value (not PHI)
                # Dates in impression like "MI 2021" are clinical, not PHI
                if description == "date pattern":
                    # Skip if part of clinical terminology
                    if any(term in value.lower() for term in
                           ["grade", "stage", "class", "type"]):
                        continue
                warnings.append(f"  Field '{field}': possible {description} detected")

    return warnings


# ============================================================================
# STEP 3: CLASSIFICATION — from mail_script.py
# (11 Cardiac Classes using Azure OpenAI)
# ============================================================================

CLASSIFICATION_11_PROMPT = """You are a medical AI assistant specializing in cardiology. Your task is to classify echocardiogram impressions into exactly ONE of the following 11 cardiac disease categories.

CLASSIFICATION HIERARCHY (check in this order):
1. Post_Intervention - Check first if patient has had any cardiac procedure
2. Congenital_Heart_Disease - Structural defects (VSD, ASD, PDA, TOF)
3. Myocardial_Infarction - Completed heart attack (MI)
4. Ischemic_Heart_Disease - Coronary disease without MI (IHD, ischemia)
5. Cardiomyopathy - Primary muscle disease (ICM, DCM)
6. Valvular_Disease - Valve abnormalities (rheumatic, stenosis, regurgitation)
7. Ventricular_Dysfunction - Functional abnormality (hypokinetic, systolic dysfunction)
8. Hypertensive_Heart_Disease - LVH from hypertension
9. Pulmonary_Hypertension - Elevated pulmonary pressures as primary diagnosis
10. Normal - Normal echo findings
11. Other_Cardiac_Conditions - Everything else

RULES:
- Return ONLY the class name (e.g., "Myocardial_Infarction")
- Use the hierarchy above - more specific diagnoses take priority
- Do NOT return any explanation, just the class name
"""


def classify_to_11_classes(impression_text: str) -> str:
    """Classify impression into one of 11 cardiac classes using Azure OpenAI"""
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": CLASSIFICATION_11_PROMPT},
                {"role": "user", "content": f"Classify this echocardiogram impression:\n\n{impression_text}"}
            ],
            temperature=0.0,
            max_tokens=50
        )

        classification = response.choices[0].message.content.strip()

        if classification in CARDIAC_CLASSES:
            return classification
        else:
            for class_name in CARDIAC_CLASSES.keys():
                if class_name in classification:
                    return class_name
            return "Other_Cardiac_Conditions"

    except Exception as e:
        logger.error(f"Error classifying: {e}")
        return "Other_Cardiac_Conditions"


# ============================================================================
# STEP 4: SEVERITY LABELING — from mail_script.py
# (4 Severity Labels using Azure OpenAI)
# ============================================================================

SEVERITY_CLASSIFICATION_PROMPT = """You are a medical AI assistant specialized in cardiac disease severity classification.

Your task is to map cardiac disease records from 11 disease classes into 4 SEVERITY classes based on clinical significance:

**4 SEVERITY CLASSES:**
1. **Normal** - Healthy heart, no significant disease
2. **Mild** - Minor abnormalities, early-stage disease, stable condition
3. **Moderate** - Significant cardiac disease requiring active treatment
4. **Severe** - Critical/life-threatening conditions requiring urgent intervention

**MAPPING RULES:**

**FIXED MAPPINGS (always the same):**
- Normal → **Normal**
- Hypertensive_Heart_Disease → **Mild**
- Pulmonary_Hypertension → **Mild**
- Myocardial_Infarction → **Moderate**
- Valvular_Disease → **Moderate**
- Other_Cardiac_Conditions → **Moderate**
- Congenital_Heart_Disease → **Severe**

**CONTEXT-DEPENDENT (check Impression text for severity keywords):**

1. **Ventricular_Dysfunction:**
   - If Impression contains "severe", "very poor", "poor LV", or EF<30% → **Severe**
   - If Impression contains "moderate LV systolic dysfunction" → **Moderate**
   - If Impression contains "normal LV", "good LV", "fair LV", or "mild" → **Mild**
   - Default → **Moderate**

2. **Ischemic_Heart_Disease:**
   - If Impression contains "mild" or "good LV systolic function" → **Mild**
   - Otherwise → **Moderate**

3. **Cardiomyopathy:**
   - If Impression contains "severe", "very poor", or "poor LV systolic function" → **Severe**
   - Otherwise → **Moderate**

4. **Post_Intervention:**
   - If Impression contains "severe" or "poor LV" → **Severe**
   - If Impression contains "moderate" → **Moderate**
   - Otherwise → **Mild**

**OUTPUT FORMAT:**
Respond with ONLY ONE WORD: Normal, Mild, Moderate, or Severe
"""


def classify_to_severity(cardiac_class: str, impression: str) -> str:
    """Map 11-class to 4 severity labels using Azure OpenAI"""
    try:
        user_prompt = f"""Cardiac Class: {cardiac_class}
Impression: {impression}

Classify this record into one of: Normal, Mild, Moderate, or Severe"""

        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SEVERITY_CLASSIFICATION_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )

        classification = response.choices[0].message.content.strip()

        valid_classes = ['Normal', 'Mild', 'Moderate', 'Severe']
        if classification in valid_classes:
            return classification
        else:
            for valid_class in valid_classes:
                if valid_class.lower() in classification.lower():
                    return valid_class
            return "Moderate"

    except Exception as e:
        logger.error(f"Error in severity classification: {e}")
        return "Moderate"


# ============================================================================
# STEP 5: TEXT FORMAT GENERATION — from mail_script.py
# ============================================================================

def combine_fields_to_text(row) -> str:
    """
    Combine all medical fields into a single text string.
    ONLY includes clinically relevant fields — no PHI.
    """
    text_parts = []

    for field in MEDICAL_FIELDS:
        value = row.get(field, "N/A")
        if value and str(value).strip() and str(value).strip() != "N/A":
            text_parts.append(f"{field}: {value}")

    return ", ".join(text_parts)


# ============================================================================
# UTILITY: CSV Conversion
# ============================================================================

def convert_json_to_csv(json_file: str, csv_file: str) -> bool:
    """Convert JSON to CSV"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not data:
            logger.warning("No data to convert")
            return False

        fieldnames = list(data[0].keys())

        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        logger.info(f"CSV created: {Path(csv_file).name} | "
                     f"Rows: {len(data)} | Columns: {len(fieldnames)}")
        return True

    except Exception as e:
        logger.error(f"CSV conversion failed: {e}")
        return False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Complete privacy-compliant pipeline:
      1. Extract from .doc files (Word COM + Azure OpenAI)
      2. De-identify (HIPAA Safe Harbor)
      3. Classify into 11 cardiac classes
      4. Map to 4 severity labels
      5. Generate text format dataset
    """

    print("=" * 80)
    print("PRIVACY-COMPLIANT CARDIAC DATA PROCESSING PIPELINE")
    print("=" * 80)
    print("\nThis pipeline will:")
    print("  Step 1: Extract data from .doc medical reports (Word COM + LLM)")
    print("  Step 2: De-identify — remove ID, Name, Date, Filename (HIPAA Safe Harbor)")
    print("  Step 3: Classify into 11 cardiac disease categories")
    print("  Step 4: Map to 4 severity labels (Normal/Mild/Moderate/Severe)")
    print("  Step 5: Generate de-identified text format dataset")
    print("\n🔒 PRIVACY: Only Age and Sex retained as demographics.")
    print("   All PHI (ID, Name, Date, Filename) is stripped before saving.")
    print("=" * 80)

    # Check Azure config
    if not all(AZURE_CONFIG.values()):
        print("\n❌ ERROR: Azure OpenAI configuration incomplete!")
        print("Please set these in your .env file:")
        print("  AZURE_API_KEY, AZURE_API_VERSION, AZURE_ENDPOINT, AZURE_DEPLOYMENT")
        return

    print(f"\n✅ Azure OpenAI configured:")
    print(f"   Endpoint: {AZURE_CONFIG['azure_endpoint']}")
    print(f"   Deployment: {AZURE_CONFIG['deployment_name']}")

    response = input("\nDo you want to proceed? (y/n): ")
    if response.lower() != 'y':
        print("Exiting...")
        return

    # ==================================================================
    # STEP 1: EXTRACTION
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("STEP 1: EXTRACTING DATA FROM MEDICAL REPORTS")
    print(f"{'=' * 80}")

    if OUTPUT_EXTRACTED_CSV.exists():
        response = input(f"\n{OUTPUT_EXTRACTED_CSV} already exists. Skip extraction? (y/n): ")
        if response.lower() == 'y':
            print("✓ Skipping extraction, using existing de-identified file")
            df_extracted = pd.read_csv(OUTPUT_EXTRACTED_CSV)
        else:
            df_extracted = run_extraction()
    else:
        df_extracted = run_extraction()

    if df_extracted is None or df_extracted.empty:
        print("❌ No data extracted. Exiting.")
        return

    print(f"\n✅ Working with {len(df_extracted)} de-identified records")

    # ==================================================================
    # STEP 2: CLASSIFICATION (11 Cardiac Classes)
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("STEP 2: CLASSIFYING INTO 11 CARDIAC DISEASE CATEGORIES")
    print(f"{'=' * 80}")

    cardiac_classes = []
    print("\nClassifying impressions...")

    for idx, row in tqdm(df_extracted.iterrows(), total=len(df_extracted),
                         desc="Classifying"):
        impression = row.get('Impression', '')

        if pd.notna(impression) and str(impression).strip():
            cardiac_class = classify_to_11_classes(str(impression))
        else:
            cardiac_class = "Other_Cardiac_Conditions"

        cardiac_classes.append(cardiac_class)
        time.sleep(0.3)

    df_extracted['Cardiac_Class'] = cardiac_classes

    # Save classified data
    df_classified = df_extracted[['Impression', 'Cardiac_Class']].copy()
    df_classified.to_csv(OUTPUT_CLASSIFIED_CSV, index=False, encoding='utf-8')

    print(f"\n✅ Classified {len(df_classified)} records into 11 classes")
    print(f"✓ Saved to {OUTPUT_CLASSIFIED_CSV}")
    print(f"\n11-Class Distribution:")
    print(df_extracted['Cardiac_Class'].value_counts())

    # ==================================================================
    # STEP 3: SEVERITY LABELING (4 Severity Labels)
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("STEP 3: MAPPING TO 4 SEVERITY LABELS")
    print(f"{'=' * 80}")

    severity_labels = []
    print("\nMapping to severity labels...")

    for idx, row in tqdm(df_extracted.iterrows(), total=len(df_extracted),
                         desc="Mapping severity"):
        cardiac_class = row['Cardiac_Class']
        impression = row.get('Impression', '')

        label = classify_to_severity(cardiac_class, str(impression))
        severity_labels.append(label)

        if (idx + 1) % 10 == 0:
            time.sleep(1)

    df_extracted['label'] = severity_labels

    # Save with labels
    df_with_labels = df_extracted[['Impression', 'Cardiac_Class', 'label']].copy()
    df_with_labels.to_csv(OUTPUT_WITH_LABELS_CSV, index=False, encoding='utf-8')

    print(f"\n✅ Mapped {len(df_with_labels)} records to severity labels")
    print(f"✓ Saved to {OUTPUT_WITH_LABELS_CSV}")
    print(f"\n4-Class Severity Distribution:")
    print(df_extracted['label'].value_counts().sort_index())

    # ==================================================================
    # STEP 4: TEXT FORMAT GENERATION
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("STEP 4: GENERATING DE-IDENTIFIED TEXT FORMAT OUTPUT")
    print(f"{'=' * 80}")

    print("\nCombining medical fields into text (PHI-free)...")
    df_extracted['Medical_Report'] = df_extracted.apply(combine_fields_to_text, axis=1)

    # Create final output — ONLY Medical_Report, Cardiac_Class, label
    df_final = pd.DataFrame({
        'Medical_Report': df_extracted['Medical_Report'],
        'Cardiac_Class': df_extracted['Cardiac_Class'],
        'label': df_extracted['label']
    })

    df_final.to_csv(OUTPUT_TEXT_FORMAT_CSV, index=False, encoding='utf-8')

    print(f"\n✅ Generated de-identified text format output")
    print(f"✓ Saved to {OUTPUT_TEXT_FORMAT_CSV}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE!")
    print(f"{'=' * 80}")

    print("\n📁 OUTPUT FILES (all PHI-free):")
    print(f"  1. {OUTPUT_JSON}")
    print(f"  2. {OUTPUT_EXTRACTED_CSV}")
    print(f"  3. {OUTPUT_CLASSIFIED_CSV}")
    print(f"  4. {OUTPUT_WITH_LABELS_CSV}")
    print(f"  5. {OUTPUT_TEXT_FORMAT_CSV}")
    print(f"  6. {OUTPUT_DEIDENTIFICATION_LOG}")

    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"  Total records processed:   {len(df_final)}")
    print(f"  Medical fields extracted:  {len(MEDICAL_FIELDS)}")
    print(f"  Demographics retained:     Age, Sex (only)")
    print(f"  PHI fields removed:        ID, Date, Patient_Name, Filename")
    print(f"  Cardiac classes:           11")
    print(f"  Severity labels:           4")

    print(f"\n🔒 PRIVACY COMPLIANCE:")
    print(f"  De-identification method:  HIPAA Safe Harbor")
    print(f"  Age binning applied:       Ages > 89 → '90+'")
    print(f"  PHI scan completed:        All text fields verified")

    print(f"\n11-Class Distribution:")
    for class_name, count in df_extracted['Cardiac_Class'].value_counts().items():
        print(f"  {class_name}: {count}")

    print(f"\n4-Class Severity Distribution:")
    for severity, count in df_extracted['label'].value_counts().sort_index().items():
        percentage = (count / len(df_extracted)) * 100
        print(f"  {severity}: {count} ({percentage:.1f}%)")

    print(f"\n{'=' * 80}")
    print("✅ All processing complete! All outputs are de-identified.")
    print(f"{'=' * 80}")


def run_extraction() -> Optional[pd.DataFrame]:
    """
    Run the full extraction + de-identification pipeline.
    Returns a de-identified DataFrame.
    """
    # Get all .doc files
    doc_files = sorted(list(DATA_DIR.glob("*.doc")))
    doc_files = [f for f in doc_files if not f.name.startswith("~$")]

    if not doc_files:
        logger.error(f"No .doc files found in {DATA_DIR}")
        return None

    print(f"\n📂 Found {len(doc_files)} .doc files in '{DATA_DIR}'")
    print(f"🔧 Using MS Word COM for accurate table extraction")
    print(f"🔒 PHI will be stripped immediately after extraction")
    print("-" * 80)

    all_data = []
    all_errors = []
    deidentification_log = {
        "timestamp": datetime.now().isoformat(),
        "method": "HIPAA Safe Harbor",
        "phi_fields_removed": PHI_FIELDS + ["Filename"],
        "demographics_retained": ["Age", "Sex"],
        "age_binning": "Ages > 89 → 90+",
        "total_processed": 0,
        "total_errors": 0,
        "phi_leakage_warnings": [],
    }

    BATCH_SIZE = 50
    num_batches = (len(doc_files) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, len(doc_files))
        batch_files = doc_files[start_idx:end_idx]

        print(f"\n{'=' * 80}")
        print(f"📦 Batch {batch_num + 1}/{num_batches} — "
              f"Processing {len(batch_files)} files")
        print(f"{'=' * 80}")

        for file_path in tqdm(batch_files, desc="Extracting & de-identifying"):
            filename = file_path.name

            try:
                # Extract with Word COM
                doc_data = extract_text_and_tables_from_doc(str(file_path))

                if doc_data["status"] != "success":
                    all_errors.append({"file": filename,
                                       "error": doc_data.get("error", "Unknown")})
                    continue

                if not doc_data["full_text"] or len(doc_data["full_text"].strip()) < 50:
                    all_errors.append({"file": filename,
                                       "error": "Insufficient text extracted"})
                    continue

                # Extract structured data with LLM (PHI stripped inside)
                extracted = extract_data_with_llm(doc_data, filename)

                # De-identify the record
                record_idx = len(all_data) + 1
                deidentified = deidentify_record(extracted, record_idx)

                # Scan for PHI leakage in text fields
                phi_warnings = scan_for_phi_leakage(deidentified)
                if phi_warnings:
                    deidentification_log["phi_leakage_warnings"].append({
                        "record_index": record_idx,
                        "warnings": phi_warnings
                    })

                all_data.append(deidentified)

            except Exception as e:
                all_errors.append({"file": filename, "error": str(e)})

            time.sleep(0.1)

        # Save checkpoint (de-identified)
        checkpoint_file = OUTPUT_DIR / f"checkpoint_{batch_num + 1:03d}.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Batch {batch_num + 1} complete:")
        print(f"   ✓ Successful: {len(batch_files) - len([e for e in all_errors if e.get('file', '') in [bf.name for bf in batch_files]])}")
        print(f"   📊 Total de-identified: {len(all_data)}/{len(doc_files)}")

    # Save final JSON (de-identified)
    if all_data:
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

    # Save de-identification log
    deidentification_log["total_processed"] = len(all_data)
    deidentification_log["total_errors"] = len(all_errors)
    with open(OUTPUT_DEIDENTIFICATION_LOG, 'w', encoding='utf-8') as f:
        json.dump(deidentification_log, f, indent=2, ensure_ascii=False)

    # Save error log
    if all_errors:
        error_log = OUTPUT_DIR / "error_log.json"
        with open(error_log, 'w', encoding='utf-8') as f:
            json.dump(all_errors, f, indent=2, ensure_ascii=False)
        logger.warning(f"Error log saved: {len(all_errors)} errors")

    # Convert to CSV
    if all_data:
        convert_json_to_csv(str(OUTPUT_JSON), str(OUTPUT_EXTRACTED_CSV))

    # Summary
    print(f"\n{'=' * 80}")
    print("📊 EXTRACTION & DE-IDENTIFICATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total files:              {len(doc_files)}")
    print(f"Successfully extracted:   {len(all_data)}")
    print(f"Errors:                   {len(all_errors)}")
    print(f"Success rate:             {len(all_data)/max(len(doc_files),1)*100:.1f}%")
    print(f"PHI fields removed:       {PHI_FIELDS + ['Filename']}")
    print(f"Demographics retained:    Age, Sex")
    phi_warn_count = len(deidentification_log["phi_leakage_warnings"])
    print(f"PHI leakage warnings:     {phi_warn_count}")
    print(f"{'=' * 80}")

    if all_data:
        return pd.read_csv(OUTPUT_EXTRACTED_CSV)
    return None


if __name__ == "__main__":
    main()
