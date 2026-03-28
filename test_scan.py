from dataset_manager import scan_dataset, parse_gender_from_filename
import pandas as pd

DATASET = r"C:\Users\onlin\Downloads\Midterm_Dataset_2026-20260328T161509Z-3-001\Midterm_Dataset_2026"

print("=== DOSYA ADANINDAN CİNSİYET PARSE TESTİ ===")
test_names = [
    "G01_D01_C_11_Angry_C3.wav",
    "G04_D01_M_20_Furious.wav",
    "G06_D01_F_20_Neutral.wav",
    "G09_D01_K_22_Happy.wav",
    "G15_D01_E_21_Notr.wav",
    "G02_D01_C_12_Mutlu_C2.wav",
]
for name in test_names:
    g = parse_gender_from_filename(name)
    print(f"  {name:<40} -> {g}")

print("\n=== VERİ SETİ TARAMA ===")
records = scan_dataset(DATASET)
df = pd.DataFrame(records)
print(f"Toplam etiketlenmiş WAV: {len(df)}")
print(f"\nSınıf dağılımı:")
print(df['gender_true'].value_counts().to_string())
print(f"\nGrup bazlı dosya sayısı:")
print(df.groupby('group').size().to_string())
