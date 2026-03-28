import zipfile
import xml.etree.ElementTree as ET
import glob
import os

dataset_path = r"C:\Users\onlin\Downloads\Midterm_Dataset_2026-20260328T161509Z-3-001\Midterm_Dataset_2026"
excel_files = glob.glob(os.path.join(dataset_path, "**", "*.xlsx"), recursive=True)

if not excel_files:
    print("No Excel files found.")
else:
    file_path = excel_files[0]
    print(f"Reading {file_path}")
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            strings_xml = z.read('xl/sharedStrings.xml')
            root = ET.fromstring(strings_xml)
            ns = {'ns': root.tag.split('}')[0].strip('{')}
            strings = [el.text for el in root.findall('.//ns:t', ns)]
            print("Shared Strings (Likely headers first):", strings[:20])
    except Exception as e:
        print("Error:", e)
