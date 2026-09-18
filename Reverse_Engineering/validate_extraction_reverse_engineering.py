"""
Reverse Engineering Validation Script
Compares extracted JSON data with re-extraction using EXACT SAME METHOD
Measures extraction consistency: Does the same method produce the same results?

Flow:
1. Load JSON record (original extraction)
2. Re-extract from .doc using EXACT SAME method (Word COM + format_tables + extract_data_with_llm)
3. Compare field-by-field: JSON vs Re-extracted
4. Calculate accuracy percentage

This validates extraction method consistency, not LLM vs document accuracy.
"""

import os
import json
import csv
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

# EXACT field names to extract (same as extraction script)
FIELD_NAMES = [
    # Patient Information
    "ID", "Date", "Patient_Name", "Age", "Sex",
    
    # Measurements (M-mode and 2-D) - with units
    "IVST", "LVIDd", "LA", "LVPWT", "LVIDs", "AO", "RVGWT", "FS", "ACS", 
    "RV", "EF", "LVEDV", "PA", "EF_Shope", "LVESV", "AV_ring", "MV_ring", "MVA",
    
    # Description (M-mode and 2-D) - LV Details
    "LV_Cavity_Size", "LV_Wall_Thickness", "LV_Wall_Motion",
    
    # Description - Other Chambers and Valves
    "RA", "MV", "RV_Description", "AV", "LA_Description", "PV", "Aorta",
    "TV", "PA_Description", "IAS", "RVOT", "IVS", "ASD", "Thrombus", 
    "VSD", "Vegetation", "PDA", "Pericardium",
    
    # Color Flow Mapping and Doppler Study
    "Mitral_Valve_Flow", "Aortic_Valve_Flow", "Pulmonary_Valve_Flow", 
    "Tricuspid_Valve_Flow", "VSD_Flow", "Others_Flow",
    
    # Impression
    "Impression"
]


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


def extract_data_with_llm(document_data: Dict, filename: str, retry_count: int = 3) -> Dict[str, str]:
    """
    EXACT SAME EXTRACTION METHOD AS extract_medical_reports_improved.py
    Use Azure OpenAI to extract structured data from document
    """
    
    full_text = document_data.get("full_text", "")
    tables = document_data.get("tables", [])
    tables_formatted = format_tables_for_llm(tables)
    
    system_prompt = """You are an expert medical data extraction specialist for Color Doppler Echocardiogram Reports.

EXTRACTION STRATEGY:

1. **Header Information** (usually at the top or in first table):
   - ID: Patient ID number
   - Patient_Name: Full name
   - Age: Just the number (e.g., "55")
   - Sex: Male/Female
   - Date: Date in format shown (e.g., "14-12-2021")

2. **Measurement Table** (look for "Measurement : (M - mode and 2 - D)"):
   - Extract measurements with units (e.g., "11 mm", "41 %", "2.4 sqcm")
   - Common fields: IVST, LVIDd, LA, LVPWT, LVIDs, AO, FS, EF, MVA, etc.
   - If a measurement cell is empty or "mm" only -> use "N/A"
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
"""

    user_prompt = f"""Extract ALL fields from this Color Doppler Echocardiogram Report.

FILENAME: {filename}

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

Return ONLY valid JSON with the required fields."""

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
            
            return result
            
        except Exception as e:
            if attempt < retry_count - 1:
                time.sleep((attempt + 1) * 2)
            else:
                print(f"⚠️  LLM extraction failed for {filename}: {str(e)}")
                return {field: "N/A" for field in FIELD_NAMES}


def compare_extractions(json_data: Dict, re_extracted_data: Dict, filename: str) -> Dict:
    """
    Compare JSON data (original extraction) with re-extracted data (same method)
    Returns field-by-field comparison with accuracy
    """
    
    # Remove Filename from both
    json_fields = {k: v for k, v in json_data.items() if k != 'Filename'}
    re_extracted_fields = {k: v for k, v in re_extracted_data.items() if k != 'Filename'}
    
    validations = {}
    total_fields = len(FIELD_NAMES)
    correct = 0
    incorrect = 0
    partial = 0
    
    for field in FIELD_NAMES:
        json_value = json_fields.get(field, "N/A")
        re_extracted_value = re_extracted_fields.get(field, "N/A")
        
        # Normalize values for comparison
        json_norm = str(json_value).strip().lower()
        re_extracted_norm = str(re_extracted_value).strip().lower()
        
        if json_norm == re_extracted_norm:
            status = "CORRECT"
            correct += 1
        elif json_norm in re_extracted_norm or re_extracted_norm in json_norm:
            status = "PARTIAL"
            partial += 1
        else:
            status = "INCORRECT"
            incorrect += 1
        
        validations[field] = {
            "status": status,
            "json_value": json_value,
            "re_extracted_value": re_extracted_value
        }
    
    accuracy_percentage = (correct + (partial * 0.5)) / total_fields * 100 if total_fields > 0 else 0
    
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


def validate_all_records(
    json_file: str,
    dataset_folder: str,
    output_csv: str,
    output_json: str
):
    """
    Main validation function - validates all records
    """
    
    print("=" * 80)
    print("REVERSE ENGINEERING VALIDATION")
    print("Comparing Extracted Data vs Original Documents")
    print("=" * 80)
    
    # Load extracted data
    print(f"\n📂 Loading extracted data: {Path(json_file).name}")
    with open(json_file, 'r', encoding='utf-8') as f:
        extracted_records = json.load(f)
    
    print(f"✅ Loaded {len(extracted_records)} extracted records")
    
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
        filename = record.get("Filename", "")
        
        if not filename:
            print(f"\n⚠️  Record {idx+1}: No filename, skipping")
            csv_summary.append({
                "Record_Index": idx,
                "Filename": "MISSING",
                "Status": "NO_FILENAME",
                "Accuracy_%": 0,
                "Correct_Fields": 0,
                "Total_Fields": 0,
                "Error": "No filename in extracted data"
            })
            total_errors += 1
            continue
        
        # Find original file
        original_file_path = Path(dataset_folder) / filename
        
        if not original_file_path.exists():
            print(f"\n⚠️  Record {idx+1}: File not found: {filename}")
            csv_summary.append({
                "Record_Index": idx,
                "Filename": filename,
                "Status": "FILE_NOT_FOUND",
                "Accuracy_%": 0,
                "Correct_Fields": 0,
                "Total_Fields": 0,
                "Error": "Original file not found"
            })
            total_errors += 1
            continue
        
        # Extract from original document
        try:
            original_doc = extract_text_and_tables_from_doc(str(original_file_path))
            
            if original_doc["status"] != "success":
                print(f"\n⚠️  Record {idx+1}: Failed to extract: {filename}")
                csv_summary.append({
                    "Record_Index": idx,
                    "Filename": filename,
                    "Status": "EXTRACTION_ERROR",
                    "Accuracy_%": 0,
                    "Correct_Fields": 0,
                    "Total_Fields": 0,
                    "Error": original_doc.get("error", "Unknown error")
                })
                total_errors += 1
                continue
            
            # Re-extract using EXACT SAME METHOD
            re_extracted_data = extract_data_with_llm(original_doc, filename)
            
            # Compare JSON vs Re-extracted
            validation = compare_extractions(record, re_extracted_data, filename)
            
            if "error" in validation:
                print(f"\n⚠️  Record {idx+1}: Validation error: {filename}")
                csv_summary.append({
                    "Record_Index": idx,
                    "Filename": filename,
                    "Status": "VALIDATION_ERROR",
                    "Accuracy_%": 0,
                    "Correct_Fields": 0,
                    "Total_Fields": 0,
                    "Error": validation["error"]
                })
                total_errors += 1
                continue
            
            # Success - store results
            summary = validation.get("summary", {})
            
            csv_summary.append({
                "Record_Index": idx,
                "Filename": filename,
                "Status": "VALIDATED",
                "Accuracy_%": summary.get("accuracy_percentage", 0),
                "Correct_Fields": summary.get("correct", 0),
                "Total_Fields": summary.get("total_fields", 0),
                "Incorrect_Fields": summary.get("incorrect", 0),
                "Partial_Fields": summary.get("partial", 0),
                "Error": ""
            })
            
            validation_results.append({
                "record_index": idx,
                "filename": filename,
                "status": "success",
                "accuracy": summary.get("accuracy_percentage", 0),
                "summary": summary,
                "field_validations": validation.get("validations", {})
            })
            
            total_success += 1
            
        except Exception as e:
            print(f"\n❌ Record {idx+1}: Unexpected error: {str(e)}")
            csv_summary.append({
                "Record_Index": idx,
                "Filename": filename,
                "Status": "ERROR",
                "Accuracy_%": 0,
                "Correct_Fields": 0,
                "Total_Fields": 0,
                "Error": str(e)
            })
            total_errors += 1
        
        total_processed += 1
        
        # Small delay to avoid overwhelming API
        time.sleep(0.5)
    
    # Calculate overall statistics
    print(f"\n{'='*80}")
    print("VALIDATION COMPLETE")
    print(f"{'='*80}")
    print(f"\n📊 Statistics:")
    print(f"   Total records: {len(extracted_records)}")
    print(f"   Successfully validated: {total_success}")
    print(f"   Errors: {total_errors}")
    
    if total_success > 0:
        avg_accuracy = sum(r["accuracy"] for r in validation_results) / len(validation_results)
        print(f"\n🎯 AVERAGE ACCURACY: {avg_accuracy:.2f}%")
        
        # Top and bottom performers
        sorted_by_acc = sorted(validation_results, key=lambda x: x["accuracy"], reverse=True)
        
        print(f"\n✅ TOP 5 MOST ACCURATE:")
        for r in sorted_by_acc[:5]:
            print(f"   {r['filename'][:50]:50s} - {r['accuracy']:.2f}%")
        
        print(f"\n⚠️  BOTTOM 5 LEAST ACCURATE:")
        for r in sorted_by_acc[-5:]:
            print(f"   {r['filename'][:50]:50s} - {r['accuracy']:.2f}%")
    
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
            "average_accuracy": sum(r["accuracy"] for r in validation_results) / len(validation_results) if validation_results else 0,
            "json_file": json_file,
            "dataset_folder": dataset_folder
        },
        "validation_results": validation_results
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Validation complete!")
    print(f"   📊 CSV Summary: {output_csv}")
    print(f"   📄 Detailed JSON: {output_json}")
    print(f"{'='*80}")


def main():
    """Main execution"""
    
    script_dir = Path(__file__).parent
    JSON_FILE = script_dir / "extracted_medical_reports.json"
    DATASET_FOLDER = script_dir / "data set"
    OUTPUT_CSV = script_dir / "validation_accuracy_results.csv"
    OUTPUT_JSON = script_dir / "validation_detailed_report.json"
    
    if not JSON_FILE.exists():
        print(f"❌ ERROR: {JSON_FILE} not found!")
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
