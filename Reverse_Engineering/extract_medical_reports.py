"""
IMPROVED Medical Report Extraction - Accurate Table Extraction
Uses MS Word COM to properly extract structured data from .doc files
Preserves table structure for accurate field extraction
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm
import time
import win32com.client
import pythoncom
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_CONFIG = {
    "api_key": os.getenv("AZURE_API_KEY"),
    "api_version": os.getenv("AZURE_API_VERSION"),
    "azure_endpoint": os.getenv("AZURE_ENDPOINT"),
    "deployment_name": os.getenv("AZURE_DEPLOYMENT")
}

# Thread-local storage
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


# EXACT field names to extract
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


def extract_text_and_tables_from_doc(file_path: str) -> Dict[str, any]:
    """
    Extract text and tables from .doc file using MS Word COM
    Returns structured data with tables preserved
    """
    # Initialize COM for this thread
    pythoncom.CoInitialize()
    
    word = None
    doc = None
    
    try:
        # Create Word application
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0  # Disable alerts
        
        # Open document
        doc = word.Documents.Open(str(Path(file_path).absolute()))
        
        # Extract all text (paragraphs)
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
                            # Remove Word's cell markers
                            cell_text = cell_text.replace('\r\x07', '').replace('\x07', '').strip()
                            row_data.append(cell_text)
                        except:
                            row_data.append("")
                    
                    if any(cell for cell in row_data):  # Only add non-empty rows
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
        # Clean up
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


def extract_data_with_llm(document_data: Dict, filename: str, retry_count: int = 3) -> Dict[str, str]:
    """
    Use Azure OpenAI to extract structured data from document
    Now with proper table structure
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
            client = get_azure_client()
            
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


def process_single_file(file_path: Path) -> Dict:
    """Process a single .doc file"""
    filename = file_path.name
    
    try:
        # Extract document with Word COM
        doc_data = extract_text_and_tables_from_doc(str(file_path))
        
        if doc_data["status"] != "success":
            return {
                "status": "error",
                "filename": filename,
                "error": doc_data.get("error", "Unknown error"),
                "data": None
            }
        
        # Check if we got meaningful data
        if not doc_data["full_text"] or len(doc_data["full_text"].strip()) < 50:
            return {
                "status": "error",
                "filename": filename,
                "error": "Insufficient text extracted",
                "data": None
            }
        
        # Extract structured data with LLM
        extracted_data = extract_data_with_llm(doc_data, filename)
        extracted_data['Filename'] = filename
        
        return {
            "status": "success",
            "filename": filename,
            "error": None,
            "data": extracted_data
        }
        
    except Exception as e:
        return {
            "status": "error",
            "filename": filename,
            "error": str(e),
            "data": None
        }


def process_files_sequential(files: List[Path]) -> tuple:
    """
    Process files sequentially (for COM stability)
    COM objects don't work well with threading
    """
    all_data = []
    errors = []
    
    print("\n🔄 Processing files sequentially (required for Word COM)...")
    
    for file_path in tqdm(files, desc="Extracting reports"):
        result = process_single_file(file_path)
        
        if result["status"] == "success":
            all_data.append(result["data"])
        else:
            errors.append({
                "file": result["filename"],
                "error": result["error"]
            })
        
        # Small delay to avoid overwhelming Word
        time.sleep(0.1)
    
    return all_data, errors


def save_checkpoint(data: List[Dict], output_dir: str, checkpoint_num: int):
    """Save checkpoint as JSON"""
    checkpoint_file = os.path.join(output_dir, f"checkpoint_{checkpoint_num:03d}.json")
    with open(checkpoint_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 Checkpoint saved: checkpoint_{checkpoint_num:03d}.json ({len(data)} records)")


def process_all_reports(input_dir: str, output_dir: str, batch_size: int = 50):
    """
    Process all medical reports in batches
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all .doc files
    input_path = Path(input_dir)
    doc_files = sorted(list(input_path.glob("*.doc")))
    
    print(f"\n📂 Found {len(doc_files)} .doc files")
    print(f"⚙️  Batch size: {batch_size}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🔧 Using MS Word COM for accurate table extraction")
    print("-" * 80)
    
    all_data = []
    all_errors = []
    
    # Process in batches
    num_batches = (len(doc_files) + batch_size - 1) // batch_size
    
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(doc_files))
        batch_files = doc_files[start_idx:end_idx]
        
        print(f"\n{'='*80}")
        print(f"📦 Batch {batch_num + 1}/{num_batches} - Processing {len(batch_files)} files")
        print(f"{'='*80}")
        
        # Process batch sequentially (Word COM requirement)
        batch_data, batch_errors = process_files_sequential(batch_files)
        
        all_data.extend(batch_data)
        all_errors.extend(batch_errors)
        
        # Save checkpoint
        if batch_data:
            save_checkpoint(all_data, output_dir, batch_num + 1)
        
        print(f"\n✅ Batch {batch_num + 1} complete:")
        print(f"   ✓ Successful: {len(batch_data)}")
        print(f"   ✗ Errors: {len(batch_errors)}")
        print(f"   📊 Total processed: {len(all_data)}/{len(doc_files)}")
    
    # Save final JSON
    if all_data:
        final_json = os.path.join(output_dir, "extracted_medical_reports.json")
        with open(final_json, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"✅ Final JSON saved: extracted_medical_reports.json")
        print(f"   📄 Records: {len(all_data)}")
        print(f"{'='*80}")
    
    # Save error log
    if all_errors:
        error_log = os.path.join(output_dir, "error_log.json")
        with open(error_log, 'w', encoding='utf-8') as f:
            json.dump(all_errors, f, indent=2, ensure_ascii=False)
        print(f"⚠️  Error log saved: error_log.json ({len(all_errors)} errors)")
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files:            {len(doc_files)}")
    print(f"Successfully extracted: {len(all_data)}")
    print(f"Errors:                 {len(all_errors)}")
    print(f"Success rate:           {len(all_data)/len(doc_files)*100:.1f}%")
    print(f"{'='*80}")
    
    return all_data, all_errors


def convert_json_to_csv(json_file: str, csv_file: str):
    """Convert JSON to CSV"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("⚠️  No data to convert")
            return False
        
        # Get field names
        fieldnames = list(data[0].keys())
        
        # Write CSV
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"\n📊 CSV created: {Path(csv_file).name}")
        print(f"   Rows: {len(data)} | Columns: {len(fieldnames)}")
        return True
        
    except Exception as e:
        print(f"❌ CSV conversion failed: {e}")
        return False


def main():
    """Main execution"""
    
    script_dir = Path(__file__).parent
    input_dir = script_dir / "data set"
    output_dir = script_dir
    
    print("=" * 80)
    print("IMPROVED MEDICAL REPORT EXTRACTION")
    print("MS Word COM | Accurate Table Extraction | LLM Processing")
    print("=" * 80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Check Azure config
    if not all(AZURE_CONFIG.values()):
        print("\n❌ ERROR: Azure OpenAI configuration incomplete!")
        print("Please check your .env file.")
        return
    
    print(f"\n✅ Azure OpenAI configured:")
    print(f"   Endpoint: {AZURE_CONFIG['azure_endpoint']}")
    print(f"   Deployment: {AZURE_CONFIG['deployment_name']}")
    
    # Configuration
    BATCH_SIZE = 50
    
    # Process all reports
    all_data, all_errors = process_all_reports(
        str(input_dir), 
        str(output_dir), 
        BATCH_SIZE
    )
    
    # Convert to CSV
    if all_data:
        print(f"\n{'='*80}")
        print("🔄 Converting to CSV...")
        print(f"{'='*80}")
        
        json_file = output_dir / "extracted_medical_reports.json"
        csv_file = output_dir / "extracted_medical_reports.csv"
        
        convert_json_to_csv(str(json_file), str(csv_file))
    
    print("\n✨ Extraction complete!")


if __name__ == "__main__":
    main()
