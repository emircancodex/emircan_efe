import os
import glob
import openpyxl
import re

DATASET_PATH = r"C:\Users\onlin\Downloads\Midterm_Dataset_2026-20260328T161509Z-3-001\Midterm_Dataset_2026"

# ──────────────────────────────────────────────────────────────────────────────
# BÖLÜM 1: Excel metadata okuma (sütun adı tutarsızlıklarına karşı dayanıklı)
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_col(name):
    """Sütun ismini normalize et."""
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")

def _find_col(headers, candidates):
    """Normalised header listesinde aday isimlerden birini bul."""
    norm = [_normalize_col(h) for h in headers]
    for c in candidates:
        if c in norm:
            return norm.index(c)
    return None

def read_excel_metadata(xlsx_path):
    """
    Bir Excel dosyasını okur; dosya adı ve cinsiyet sütunlarını bulur.
    Returns: dict {basename_lower → 'M'/'F'/'C'}
    """
    result = {}
    try:
        wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return result

        # İlk satırı header olarak dene
        headers = rows[0]
        file_idx = _find_col(headers, ['file_name', 'file_name', 'filename', 'dosya_adi',
                                        'dosya', 'file', 'audio_file', 'file_name.'])
        gender_idx = _find_col(headers, ['gender', 'cinsiyet', 'sex', 'gen', 'cins'])

        if file_idx is None or gender_idx is None:
            # Header bulunamadı → dosya adını satır içeriğinden, cinsiyeti dosya adından parse et
            for row in rows:
                for cell in row:
                    if cell and str(cell).lower().endswith('.wav'):
                        gender = parse_gender_from_filename(str(cell))
                        if gender:
                            result[str(cell).strip().lower()] = gender
        else:
            for row in rows[1:]:
                try:
                    fname = row[file_idx]
                    gval  = row[gender_idx]
                    if fname and gval:
                        fname_str = str(fname).strip()
                        if not fname_str.lower().endswith('.wav'):
                            fname_str += '.wav'
                        gender = normalize_gender(str(gval))
                        if gender:
                            result[fname_str.lower()] = gender
                except Exception:
                    pass
        wb.close()
    except Exception as e:
        print(f"  [UYARI] Excel okunamadı: {xlsx_path} → {e}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# BÖLÜM 2: Dosya adından cinsiyet ayrıştırma (birincil yöntem)
# ──────────────────────────────────────────────────────────────────────────────

# Örnek isimler:
#   G01_D01_C_11_Angry_C3.wav   → C (Çocuk)
#   G04_D01_M_20_Furious.wav    → M (Erkek)
#   G06_D01_F_20_Neutral.wav    → F (Kadın)
#   G09_D01_K_22_Happy.wav      → K (Kadın, Türkçe)
#   G15_D01_E_21_Notr.wav       → E (Erkek, Türkçe)

_FILENAME_GENDER_RE = re.compile(
    r'[_-]([MFCEKmfcek])[_-]\d',   # G01_D01_C_11_ veya G01_D01_M_20_
    re.IGNORECASE
)

def parse_gender_from_filename(filename):
    """WAV dosya adından cinsiyet harfini ayıkla. M/F/C/E/K → 'Erkek'/'Kadın'/'Çocuk'"""
    base = os.path.basename(filename)
    m = _FILENAME_GENDER_RE.search(base)
    if m:
        return normalize_gender(m.group(1))
    return None

def normalize_gender(raw):
    """Çeşitli cinsiyet ifadelerini standartlaştır."""
    r = str(raw).strip().lower()
    if r in ('m', 'male', 'erkek', 'e'):
        return 'Erkek'
    if r in ('f', 'female', 'kadın', 'k', 'kadin', 'woman', 'kadin'):
        return 'Kadın'
    if r in ('c', 'child', 'çocuk', 'ç', 'cocuk', 'kid'):
        return 'Çocuk'
    return None


# ──────────────────────────────────────────────────────────────────────────────
# BÖLÜM 3: Veri setini tara — tüm WAV dosyalarını bul ve etiketle
# ──────────────────────────────────────────────────────────────────────────────

def scan_dataset(dataset_path):
    """
    Tüm grup klasörlerini tarar.
    Her WAV için:
      1. Dosya adından cinsiyeti parse et (birincil)
      2. Excel metadata ile doğrula / eksikse oradan al
    Returns: list of dicts with keys: path, group, gender_true
    """
    records = []
    group_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    for grp in group_dirs:
        grp_path = os.path.join(dataset_path, grp)

        # Excel metadata'yı yükle (yedek olarak)
        xlsx_files = [f for f in glob.glob(os.path.join(grp_path, "*.xlsx"))
                      if not os.path.basename(f).startswith('~$')]
        excel_map = {}
        for xf in xlsx_files:
            excel_map.update(read_excel_metadata(xf))

        # WAV dosyalarını bul — Windows'ta büyük/küçük harf duyarsız glob
        # aynı dosyayı iki kez bulabileceğinden normalize ederek tekilleştiriyoruz
        _seen = {}
        for _w in glob.glob(os.path.join(grp_path, "*.wav")) + \
                  glob.glob(os.path.join(grp_path, "*.WAV")):
            _seen[os.path.normcase(os.path.abspath(_w))] = _w
        wav_files = list(_seen.values())

        for wf in wav_files:
            basename = os.path.basename(wf)

            # 1. Dosya adından parse et
            gender = parse_gender_from_filename(basename)

            # 2. Bulunamazsa Excel'e bak
            if gender is None:
                gender = excel_map.get(basename.lower())

            if gender is None:
                # hâlâ bulunamadıysa atla (bu dosyayı işleyemeyiz)
                continue

            records.append({
                'path':  wf,
                'group': grp,
                'basename': basename,
                'gender_true': gender,
            })

    return records


# ──────────────────────────────────────────────────────────────────────────────
# BÖLÜM 4: Tam veri seti üzerinde sınıflandırma ve metrikleri hesaplama
# ──────────────────────────────────────────────────────────────────────────────

def process_dataset(dataset_path, progress_callback=None):
    """
    Tüm WAV'ları analiz eder, tahmin üretir, karşılaştırır.
    progress_callback(done, total) opsiyonel ilerleme bildirimi.
    Returns: (results_df, error_message_or_None)
    """
    import pandas as pd
    from audio_utils import load_audio, analyze_audio
    from classifier import classify_gender

    records = scan_dataset(dataset_path)
    if not records:
        return None, "Hiç etiketli WAV dosyası bulunamadı."

    results = []
    total = len(records)

    for i, rec in enumerate(records):
        if progress_callback:
            progress_callback(i, total)

        y, sr = load_audio(rec['path'])
        if y is None:
            continue

        mean_f0, mean_zcr, mean_ste, _ = analyze_audio(y, sr)
        pred = classify_gender(mean_f0)

        results.append({
            'Grup':          rec['group'],
            'Dosya':         rec['basename'],
            'Gerçek Sınıf':  rec['gender_true'],
            'Tahmin':        pred,
            'F0 (Hz)':       round(mean_f0, 2),
            'ZCR':           round(mean_zcr, 5),
            'Enerji':        round(float(mean_ste), 6),
            'Doğru mu':      rec['gender_true'] == pred,
        })

    if progress_callback:
        progress_callback(total, total)

    return pd.DataFrame(results), None


def evaluate_metrics(results_df):
    """Accuracy, Confusion Matrix ve sınıf istatistiklerini hesapla."""
    import pandas as pd
    import numpy as np

    if results_df is None or len(results_df) == 0:
        return 0, None, None

    correct = results_df['Doğru mu'].sum()
    accuracy = correct / len(results_df) * 100

    all_classes = ['Erkek', 'Kadın', 'Çocuk']
    present = [c for c in all_classes if c in results_df['Gerçek Sınıf'].values or c in results_df['Tahmin'].values]

    conf_matrix = pd.crosstab(
        results_df['Gerçek Sınıf'],
        results_df['Tahmin'],
        rownames=['Gerçek'],
        colnames=['Tahmin']
    ).reindex(index=present, columns=present, fill_value=0)

    stats_rows = []
    for cls in present:
        subset = results_df[results_df['Gerçek Sınıf'] == cls]
        f0_vals = subset['F0 (Hz)'][subset['F0 (Hz)'] > 0]
        cls_total   = len(subset)
        cls_correct = subset['Doğru mu'].sum()
        stats_rows.append({
            'Sınıf':            cls,
            'Örnek Sayısı':     cls_total,
            'Ortalama F0 (Hz)': round(f0_vals.mean(), 1) if len(f0_vals) > 0 else 0,
            'Standart Sapma':   round(f0_vals.std(), 1)  if len(f0_vals) > 1 else 0,
            'Başarı (%)':       round(cls_correct / cls_total * 100, 1) if cls_total > 0 else 0,
        })

    stats = pd.DataFrame(stats_rows)
    return accuracy, conf_matrix, stats
