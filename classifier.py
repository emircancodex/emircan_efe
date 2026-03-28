def classify_gender(f0):
    """
    Rule-based classification for Gender (Male, Female, Child) based on F0.
    Male: 85 - 180 Hz
    Female: 165 - 255 Hz
    Child: > 250 Hz
    """
    if f0 == 0:
        return "Bilinmeyen (F0 Bulunamadı)"
    
    if f0 <= 165:
        return "Erkek"
    elif f0 > 165 and f0 <= 255:
        return "Kadın"
    else:
        return "Çocuk"

def normalize_gender_label(label):
    """Normalizes dataset labels into Erkek, Kadın, Çocuk."""
    label = str(label).strip().lower()
    if label in ['m', 'male', 'erkek', 'e']:
        return "Erkek"
    elif label in ['f', 'female', 'kadın', 'k', 'kadin', 'woman']:
        return "Kadın"
    elif label in ['c', 'child', 'çocuk', 'ç', 'cocuk']:
        return "Çocuk"
    else:
        return "Bilinmeyen"
