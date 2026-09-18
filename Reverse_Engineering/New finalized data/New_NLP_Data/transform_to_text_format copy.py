"""
Transform medical reports CSV to text format
Combines all medical fields into a single Medical_Report column in free-text format
"""

import pandas as pd
from pathlib import Path
#from dotenv import load_dotenv
import os

# Load environment variables
#load_dotenv()

# File paths
INPUT_CSV = Path("/Users/naimul/My Research/Reverse_Engineering/New finalized data/extracted_medical_reports_100_percent.csv")
OUTPUT_CSV = Path("/Users/naimul/My Research/Reverse_Engineering/New finalized data/New_NLP_Data/medical_reports_100_percent_text_format.csv")

# Medical fields to combine into free text (excluding Cardiac_Class and label)
MEDICAL_FIELDS = [
    "Age", "Sex", "IVST", "LVIDd", "LA", "LVPWT", "LVIDs", "AO", "RVGWT", 
    "FS", "ACS", "RV", "EF", "LVEDV", "PA", "EF_Shope", "LVESV", "AV_ring", 
    "MV_ring", "MVA", "LV_Cavity_Size", "LV_Wall_Thickness", "LV_Wall_Motion", 
    "RA", "MV", "RV_Description", "AV", "LA_Description", "PV", "Aorta", "TV", 
    "PA_Description", "IAS", "RVOT", "IVS", "ASD", "Thrombus", "VSD", 
    "Vegetation", "PDA", "Pericardium", "Mitral_Valve_Flow", "Aortic_Valve_Flow", 
    "Pulmonary_Valve_Flow", "Tricuspid_Valve_Flow", "VSD_Flow", "Others_Flow", 
    "Impression"
]


def combine_fields_to_text(row):
    """
    Combine all medical fields into a single text string
    Format: "Field: value, Field: value, ..."
    """
    text_parts = []
    
    for field in MEDICAL_FIELDS:
        value = row.get(field, "N/A")
        # Skip empty or N/A values to make text cleaner (optional)
        if value and value != "N/A" and str(value).strip():
            text_parts.append(f"{field}: {value}")
    
    return ", ".join(text_parts)


def transform_csv():
    """
    Transform the CSV from structured format to text format
    """
    print("Medical Report CSV Transformation")
    print("=" * 50)
    
    # Read the input CSV
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found!")
        return
    
    print(f"Reading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    print(f"Found {len(df)} records")
    print(f"Original columns: {len(df.columns)}")
    
    # Create the Medical_Report column
    print("\nCombining medical fields into free text...")
    df['Medical_Report'] = df.apply(combine_fields_to_text, axis=1)
    
    # Create new dataframe with desired columns
    # Keep Cardiac_Class and label as they are
    transformed_df = pd.DataFrame({
        'Medical_Report': df['Medical_Report'],
        'Cardiac_Class': df['Cardiac_Class'],
        'label': df['label']
    })
    
    # Save to new CSV
    print(f"\nSaving transformed data to {OUTPUT_CSV}...")
    transformed_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    
    print("\n" + "=" * 50)
    print("Transformation complete!")
    print(f"Input file: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_CSV}")
    print(f"Total records: {len(transformed_df)}")
    print(f"New columns: {list(transformed_df.columns)}")
    
    # Show sample
    print("\nSample of first record:")
    print("-" * 50)
    print(f"Medical_Report: {transformed_df.iloc[0]['Medical_Report'][:200]}...")
    print(f"Cardiac_Class: {transformed_df.iloc[0]['Cardiac_Class']}")
    print(f"label: {transformed_df.iloc[0]['label']}")


def main():
    """
    Main execution
    """
    try:
        transform_csv()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
