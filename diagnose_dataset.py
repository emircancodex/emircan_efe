import os
import glob
import zipfile
import xml.etree.ElementTree as ET

DATASET = r"C:\Users\onlin\Downloads\Midterm_Dataset_2026-20260328T161509Z-3-001\Midterm_Dataset_2026"

# 1. Count wav files per group
total_wav = 0
total_xlsx = 0

group_dirs = [d for d in os.listdir(DATASET) if os.path.isdir(os.path.join(DATASET, d))]
print(f"=== TOPLAM GRUP SAYISI: {len(group_dirs)} ===\n")

missing_wav_groups = []
for grp in sorted(group_dirs):
    grp_path = os.path.join(DATASET, grp)
    wavs = glob.glob(os.path.join(grp_path, "**", "*.wav"), recursive=True) + \
           glob.glob(os.path.join(grp_path, "**", "*.WAV"), recursive=True)
    xlsxs = glob.glob(os.path.join(grp_path, "**", "*.xlsx"), recursive=True)
    total_wav += len(wavs)
    total_xlsx += len(xlsxs)
    wav_names = [os.path.basename(w) for w in wavs[:3]]
    print(f"  {grp}: {len(wavs)} WAV, {len(xlsxs)} XLSX | İlk dosyalar: {wav_names}")
    if len(wavs) == 0:
        missing_wav_groups.append(grp)

print(f"\n=== TOPLAM WAV: {total_wav} | TOPLAM XLSX: {total_xlsx} ===")
if missing_wav_groups:
    print(f"\n WAV DOSYASI OLMAYAN GRUPLAR: {missing_wav_groups}")

# 2. Inspect first Excel's structure in detail
print("\n=== İLK EXCEL DETAYLI İNCELEME ===")
all_excel = glob.glob(os.path.join(DATASET, "**", "*.xlsx"), recursive=True)
all_excel = [f for f in all_excel if not os.path.basename(f).startswith('~$')]
if all_excel:
    f = all_excel[0]
    print(f"Dosya: {f}")
    try:
        with zipfile.ZipFile(f, 'r') as z:
            strings_xml = z.read('xl/sharedStrings.xml')
            root = ET.fromstring(strings_xml)
            ns = {'ns': root.tag.split('}')[0].strip('{')}
            strings = [el.text for el in root.findall('.//ns:t', ns)]
            print(f"  Tüm paylaşılan dizeler: {strings[:50]}")
    except Exception as e:
        print(f"  Hata: {e}")

# 3. Check a few more excels (different groups) to find variation in structure
print("\n=== FARKLI GRUPLAR EXCEL BAŞLIKLARI ===")
for f in all_excel[:8]:
    grp_name = os.path.basename(os.path.dirname(f))
    try:
        with zipfile.ZipFile(f, 'r') as z:
            strings_xml = z.read('xl/sharedStrings.xml')
            root = ET.fromstring(strings_xml)
            ns = {'ns': root.tag.split('}')[0].strip('{')}
            strings = [el.text for el in root.findall('.//ns:t', ns)]
            print(f"  {grp_name}: {strings[:10]}")
    except Exception as e:
        print(f"  {grp_name}: HATA - {e}")
