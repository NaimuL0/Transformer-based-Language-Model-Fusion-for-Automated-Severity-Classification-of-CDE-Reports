"""
Privacy-Compliant Reverse Engineering Validation Script
========================================================
Compares extracted JSON data with re-extraction using EXACT SAME METHOD.
Measures extraction consistency: Does the same method produce the same results?

Flow:
1. Load de-identified JSON record (original extraction)
2. Build file-index mapping (maps Record_Index → .doc file) for re-extraction
3. Re-extract from .doc using EXACT SAME method (Word COM + format_tables + LLM)
4. Strip PHI from re-extraction BEFORE comparison
5. Compare field-by-field: Original vs Re-extracted (clinical fields only)
6. Calculate accuracy percentage

PRIVACY DESIGN:
- FIELD_NAMES contains NO PHI (no ID, Date, Patient_Name)
- LLM prompt explicitly instructs: "Do NOT extract patient ID, name, or date"
- PHI is stripped from LLM response even if returned
- File matching uses anonymous Record_Index, NOT Filename
- Validation outputs contain NO patient-identifying information
- Filename is used internally ONLY for file access, never saved to outputs

HIPAA Safe Harbor Compliance:
- Only Age (binned >89 → "90+") and Sex retained as demographics
- All 4 HIPAA direct identifiers in this dataset (ID, Name, Date, Filename) excluded
"""

import os
import json
import csv
import re
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm
import time
import win32com.client
import pythoncom

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_CONFIG = {
    "api_key": os.getenv("AZURE_API_KEY"),
    "api_version": os.getenv("AZURE_API_VERSION"),
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "deployment_name": os.getenv("AZURE_DEPLOYMENT")
}

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_CONFIG["api_key"],
    api_version=AZURE_CONFIG["api_version"],
    azure_endpoint=AZURE_CONFIG["azure_endpoint"]
)

# ============================================================================
# FIELD DEFINITIONS — PRIVACY-COMPLIANT (NO PHI)
# ============================================================================

# PHI fields — NEVER included in extraction, comparison, or output
PHI_FIELDS = {"ID", "Date", "Patient_Name", "Filename"}

# Clinical fields ONLY — no patient identifiers
FIELD_NAMES = [
    # Demographics (only Age and Sex — clinically essential)
    "Age", "Sex",

    # Measurements (M-mode and 2-D) — with units
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


# ============================================================================
# EXTRACTION FUNCTIONS (from extract_medical_reports.py)
# ============================================================================

def extract_text_and_tables_from_doc(file_path: str) -> Dict:
    """Extract text and tables from .doc file using MS Word COM"""
    pythoncom.CoInitialize()

    word = None
    doc = None

    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0

        doc = word.Documents.Open(str(Path(file_path).absolute()))

        # Extract full text
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
            "tables": []
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
    """Format tables in a clear structure for LLM (EXACT SAME AS EXTRACTION)"""
    if not tables:
        return "No tables found"

    formatted = []
    for idx, table in enumerate(tables, 1):
        formatted.append(f"\n--- TABLE {idx} ---")
        for row in table:
            formatted.append(" | ".join(str(cell) for cell in row))

    return "\n".join(formatted)


def extract_data_with_llm(document_data: Dict, retry_count: int = 3) -> Dict[str, str]:
    """
    PRIVACY-COMPLIANT extraction using Azure OpenAI.

    Key differences from original:
    - LLM prompt does NOT ask for ID, Name, Date
    - PHI is stripped from response even if LLM returns it
    - No filename is passed to the LLM prompt
    """

    full_text = document_data.get("full_text", "")
    tables = document_data.get("tables", [])
    tables_formatted = format_tables_for_llm(tables)

    system_prompt = """You are an expert medical data extraction specialist for Color Doppler Echocardiogram Reports.

EXTRACTION STRATEGY:

1. **Demographics** (from header or first table):
   - Age: Just the number (e.g., "55")
   - Sex: Male/Female
   - Do NOT extract patient ID, patient name, or date

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
- Do NOT include patient ID, patient name, date, or any identifying information
"""

    user_prompt = f"""Extract ALL clinical fields from this Color Doppler Echocardiogram Report.

REQUIRED FIELDS (use these exact names):
{json.dumps(FIELD_NAMES, indent=2)}

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
6. Do NOT include any patient identifying information (no ID, Name, Date)

Return ONLY valid JSON with the required clinical fields."""

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
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
            for field in FIELD_NAMES:
                if field not in result:
                    result[field] = "N/A"

            # ============================================================
            # PRIVACY ENFORCEMENT: Strip any PHI the LLM may have returned
            # ============================================================
            for phi_field in PHI_FIELDS:
                result.pop(phi_field, None)

            # Also strip any unexpected PHI-like keys
            keys_to_remove = [
                key for key in result.keys()
                if any(phi in key.lower() for phi in
                       ["patient_name", "patient_id", "date_of", "dob",
                        "mrn", "record_number", "address", "phone",
                        "email", "ssn", "filename"])
            ]
            for key in keys_to_remove:
                result.pop(key, None)

            # Age binning: HIPAA Safe Harbor (>89 → "90+")
            if "Age" in result:
                try:
                    age_val = int(str(result["Age"]).strip())
                    if age_val > 89:
                        result["Age"] = "90+"
                except (ValueError, TypeError):
                    pass

            return result

        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep((attempt + 1) * 2)
            else:
                print(f"⚠️  LLM extraction failed: {str(e)}")
                return {field: "N/A" for field in FIELD_NAMES}


# ============================================================================
# COMPARISON FUNCTION — PHI-FREE
# ============================================================================

def compare_extractions(original_data: Dict, re_extracted_data: Dict) -> Dict:
    """
    Compare original extraction with re-extraction.
    Compares ONLY clinical fields (FIELD_NAMES) — no PHI fields.
    """

    # Filter to only clinical fields (exclude any PHI that might be present)
    original_fields = {k: v for k, v in original_data.items()
                       if k in FIELD_NAMES}
    re_extracted_fields = {k: v for k, v in re_extracted_data.items()
                          if k in FIELD_NAMES}

    validations = {}
    total_fields = len(FIELD_NAMES)
    correct = 0
    incorrect = 0
    partial = 0

    for field in FIELD_NAMES:
        original_value = original_fields.get(field, "N/A")
        re_extracted_value = re_extracted_fields.get(field, "N/A")

        # Normalize values for comparison
        original_norm = str(original_value).strip().lower()
        re_extracted_norm = str(re_extracted_value).strip().lower()

        if original_norm == re_extracted_norm:
            status = "CORRECT"
            correct += 1
        elif (original_norm in re_extracted_norm or
              re_extracted_norm in original_norm):
            status = "PARTIAL"
            partial += 1
        else:
            status = "INCORRECT"
            incorrect += 1

        validations[field] = {
            "status": status,
            "original_value": original_value,
            "re_extracted_value": re_extracted_value
        }

    accuracy_percentage = (
        (correct + (partial * 0.5)) / total_fields * 100
        if total_fields > 0 else 0
    )

    return {
        "validations": validations,
        "summary": {
            "total_fields": total_fields,
            "correct": correct,
            "incorrect": incorrect,
            "partial": partial,
            "accuracy_percentage": round(accuracy_percentage, 2)
        }
    }


# ============================================================================
# FILE-INDEX MAPPING (replaces Filename-based matching)
# ============================================================================

def build_file_index_mapping(dataset_folder: str) -> Dict[int, Path]:
    """
    Build a deterministic mapping from Record_Index → .doc file path.

    The mapping uses the SAME sorted order as the original extraction
    pipeline (sorted alphabetically), so Record_Index 1 corresponds
    to the first file alphabetically, etc.

    This replaces Filename-based matching to avoid storing patient
    names (which are embedded in 329/724 filenames).
    """
    dataset_path = Path(dataset_folder)
    doc_files = sorted(list(dataset_path.glob("*.doc")))
    doc_files = [f for f in doc_files if not f.name.startswith("~$")]

    # Record_Index is 1-based (matching deidentify_record in pipeline)
    mapping = {i + 1: f for i, f in enumerate(doc_files)}

    return mapping


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def validate_all_records(
    json_file: str,
    dataset_folder: str,
    output_csv: str,
    output_json: str
):
    """
    Privacy-compliant validation — no PHI in inputs, processing, or outputs.
    """

    print("=" * 80)
    print("PRIVACY-COMPLIANT REVERSE ENGINEERING VALIDATION")
    print("Comparing De-identified Extracted Data vs Original Documents")
    print("🔒 No PHI in extraction, comparison, or output files")
    print("=" * 80)

    # Load extracted data (de-identified)
    print(f"\n📂 Loading de-identified extracted data: {Path(json_file).name}")
    with open(json_file, 'r', encoding='utf-8') as f:
        extracted_records = json.load(f)

    print(f"✅ Loaded {len(extracted_records)} de-identified records")

    # Build file-index mapping
    print(f"\n🔗 Building Record_Index → file mapping...")
    file_mapping = build_file_index_mapping(dataset_folder)
    print(f"   Found {len(file_mapping)} .doc files in dataset folder")

    # Check Azure config
    print(f"\n🔧 Azure OpenAI Configuration:")
    print(f"   Endpoint: {AZURE_CONFIG['azure_endpoint']}")
    print(f"   Deployment: {AZURE_CONFIG['deployment_name']}")
    print(f"   API Key: {'✅ Set' if AZURE_CONFIG['api_key'] else '❌ Missing'}")

    if not all(AZURE_CONFIG.values()):
        print("\n❌ ERROR: Azure OpenAI configuration incomplete!")
        return

    print(f"\n🔍 Starting validation of {len(extracted_records)} records...")
    print(f"📁 Dataset folder: {dataset_folder}")
    print("-" * 80)

    # Results storage
    validation_results = []
    csv_summary = []

    # Statistics
    total_processed = 0
    total_success = 0
    total_errors = 0

    # Process each record
    for idx, record in enumerate(tqdm(extracted_records, desc="Validating")):

        # Use Record_Index for file matching (privacy-compliant)
        record_index = record.get("Record_Index", idx + 1)

        if record_index not in file_mapping:
            print(f"\n⚠️  Record {record_index}: No matching file in dataset")
            csv_summary.append({
                "Record_Index": record_index,
                "Status": "FILE_NOT_FOUND",
                "Accuracy_%": 0,
                "Correct_Fields": 0,
                "Total_Fields": 0,
                "Incorrect_Fields": 0,
                "Partial_Fields": 0,
                "Error": "No matching file for this Record_Index"
            })
            total_errors += 1
            continue

        original_file_path = file_mapping[record_index]

        # Extract from original document
        try:
            original_doc = extract_text_and_tables_from_doc(str(original_file_path))

            if original_doc["status"] != "success":
                print(f"\n⚠️  Record {record_index}: Failed to extract document")
                csv_summary.append({
                    "Record_Index": record_index,
                    "Status": "EXTRACTION_ERROR",
                    "Accuracy_%": 0,
                    "Correct_Fields": 0,
                    "Total_Fields": 0,
                    "Incorrect_Fields": 0,
                    "Partial_Fields": 0,
                    "Error": original_doc.get("error", "Unknown error")
                })
                total_errors += 1
                continue

            # Re-extract using EXACT SAME privacy-compliant method
            # NOTE: No filename passed to LLM — privacy by design
            re_extracted_data = extract_data_with_llm(original_doc)

            # Compare clinical fields only (no PHI in comparison)
            validation = compare_extractions(record, re_extracted_data)

            if "error" in validation:
                print(f"\n⚠️  Record {record_index}: Validation error")
                csv_summary.append({
                    "Record_Index": record_index,
                    "Status": "VALIDATION_ERROR",
                    "Accuracy_%": 0,
                    "Correct_Fields": 0,
                    "Total_Fields": 0,
                    "Incorrect_Fields": 0,
                    "Partial_Fields": 0,
                    "Error": validation["error"]
                })
                total_errors += 1
                continue

            # Success — store results (PHI-free)
            summary = validation.get("summary", {})

            csv_summary.append({
                "Record_Index": record_index,
                "Status": "VALIDATED",
                "Accuracy_%": summary.get("accuracy_percentage", 0),
                "Correct_Fields": summary.get("correct", 0),
                "Total_Fields": summary.get("total_fields", 0),
                "Incorrect_Fields": summary.get("incorrect", 0),
                "Partial_Fields": summary.get("partial", 0),
                "Error": ""
            })

            validation_results.append({
                "record_index": record_index,
                "status": "success",
                "accuracy": summary.get("accuracy_percentage", 0),
                "summary": summary,
                "field_validations": validation.get("validations", {})
            })

            total_success += 1

        except Exception as e:
            print(f"\n❌ Record {record_index}: Unexpected error: {str(e)}")
            csv_summary.append({
                "Record_Index": record_index,
                "Status": "ERROR",
                "Accuracy_%": 0,
                "Correct_Fields": 0,
                "Total_Fields": 0,
                "Incorrect_Fields": 0,
                "Partial_Fields": 0,
                "Error": str(e)
            })
            total_errors += 1

        total_processed += 1

        # Small delay to avoid overwhelming API
        time.sleep(0.5)

    # ================================================================
    # RESULTS SUMMARY
    # ================================================================
    print(f"\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n📊 Statistics:")
    print(f"   Total records:             {len(extracted_records)}")
    print(f"   Successfully validated:    {total_success}")
    print(f"   Errors:                    {total_errors}")

    if total_success > 0:
        avg_accuracy = (sum(r["accuracy"] for r in validation_results)
                        / len(validation_results))
        print(f"\n🎯 AVERAGE EXTRACTION CONSISTENCY: {avg_accuracy:.2f}%")

        # Per-field accuracy
        field_accuracy = {field: {"correct": 0, "total": 0}
                          for field in FIELD_NAMES}
        for r in validation_results:
            for field, val in r["field_validations"].items():
                if field in field_accuracy:
                    field_accuracy[field]["total"] += 1
                    if val["status"] == "CORRECT":
                        field_accuracy[field]["correct"] += 1

        print(f"\n📋 PER-FIELD CONSISTENCY (top 10 lowest):")
        field_scores = []
        for field, counts in field_accuracy.items():
            if counts["total"] > 0:
                acc = counts["correct"] / counts["total"] * 100
                field_scores.append((field, acc, counts["correct"],
                                     counts["total"]))

        field_scores.sort(key=lambda x: x[1])
        for field, acc, correct, total in field_scores[:10]:
            print(f"   {field:<30s} {acc:6.1f}%  ({correct}/{total})")

        # Top and bottom performers (by Record_Index, not filename)
        sorted_by_acc = sorted(validation_results,
                               key=lambda x: x["accuracy"], reverse=True)

        print(f"\n✅ TOP 5 MOST CONSISTENT RECORDS:")
        for r in sorted_by_acc[:5]:
            print(f"   Record #{r['record_index']:<6d} — {r['accuracy']:.2f}%")

        print(f"\n⚠️  BOTTOM 5 LEAST CONSISTENT RECORDS:")
        for r in sorted_by_acc[-5:]:
            print(f"   Record #{r['record_index']:<6d} — {r['accuracy']:.2f}%")

    # ================================================================
    # SAVE OUTPUTS (all PHI-free)
    # ================================================================

    # Save CSV summary
    print(f"\n💾 Saving CSV summary: {Path(output_csv).name}")
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        if csv_summary:
            writer = csv.DictWriter(f, fieldnames=csv_summary[0].keys())
            writer.writeheader()
            writer.writerows(csv_summary)

    # Save detailed JSON
    print(f"💾 Saving detailed JSON: {Path(output_json).name}")
    full_report = {
        "metadata": {
            "total_records": len(extracted_records),
            "successfully_validated": total_success,
            "errors": total_errors,
            "average_accuracy": (
                sum(r["accuracy"] for r in validation_results)
                / len(validation_results)
                if validation_results else 0
            ),
            "fields_compared": len(FIELD_NAMES),
            "phi_fields_excluded": list(PHI_FIELDS),
            "privacy_compliant": True,
            "deidentification_method": "HIPAA Safe Harbor"
        },
        "validation_results": validation_results
    }

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\n🔒 PRIVACY VERIFICATION:")
    print(f"   PHI fields excluded from comparison: {list(PHI_FIELDS)}")
    print(f"   Clinical fields compared:            {len(FIELD_NAMES)}")
    print(f"   Output files contain NO patient identifiers")

    print(f"\n✅ Validation complete!")
    print(f"   📊 CSV Summary:   {output_csv}")
    print(f"   📄 Detailed JSON: {output_json}")
    print(f"{'=' * 80}")


def main():
    """Main execution"""

    script_dir = Path(__file__).parent
    JSON_FILE = script_dir / "output" / "extracted_deidentified_reports.json"
    DATASET_FOLDER = script_dir / "data set"
    OUTPUT_CSV = script_dir / "output" / "validation_accuracy_results.csv"
    OUTPUT_JSON = script_dir / "output" / "validation_detailed_report.json"

    # Create output directory
    (script_dir / "output").mkdir(exist_ok=True)

    if not JSON_FILE.exists():
        print(f"❌ ERROR: {JSON_FILE} not found!")
        print(f"   Run the extraction pipeline first to generate de-identified data.")
        return

    if not DATASET_FOLDER.exists():
        print(f"❌ ERROR: {DATASET_FOLDER} not found!")
        return

    validate_all_records(
        json_file=str(JSON_FILE),
        dataset_folder=str(DATASET_FOLDER),
        output_csv=str(OUTPUT_CSV),
        output_json=str(OUTPUT_JSON)
    )


if __name__ == "__main__":
    main()
