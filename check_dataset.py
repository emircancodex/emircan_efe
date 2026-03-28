import os
import glob
import pandas as pd

dataset_path = r"C:\Users\onlin\Downloads\Midterm_Dataset_2026-20260328T161509Z-3-001\Midterm_Dataset_2026"
excel_files = glob.glob(os.path.join(dataset_path, "**", "*.xlsx"), recursive=True)

if not excel_files:
    print("No Excel files found.")
else:
    print(f"Found {len(excel_files)} Excel files. Inspecting the first one: {excel_files[0]}")
    try:
        df = pd.read_excel(excel_files[0])
        print("Columns:", df.columns.tolist())
        print("First row data:")
        print(df.head(1).to_dict('records'))
    except Exception as e:
        print("Error reading excel file:", e)
