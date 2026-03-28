import numpy as np
import librosa
import scipy.signal

def load_audio(file_path, sr=16000):
    """Loads an audio file and resamples it to 'sr'."""
    try:
        y, sr = librosa.load(file_path, sr=sr)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def calculate_ste_zcr(y, frame_length, hop_length):
    """Calculates Short-Time Energy and Zero Crossing Rate."""
    if len(y) == 0:
        return np.array([]), np.array([])
    ste = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)])
    # Ensure zero crossing is calculated per frame with similar properties
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    return ste, zcr

def compute_autocorrelation_pitch(y, sr, fmin=75, fmax=500):
    """Computes F0 using the Autocorrelation technique for a single frame."""
    if len(y) == 0:
        return 0, np.array([])
    
    # Compute auto-correlation using scipy.signal.correlate
    autocorr = scipy.signal.correlate(y, y, mode='full')
    autocorr = autocorr[len(autocorr)//2:] # Keep positive lags
    
    if len(autocorr) == 0:
         return 0, np.array([])
         
    # Restrict lags based on physiological pitch ranges (fmin, fmax)
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    
    if max_lag >= len(autocorr):
        max_lag = len(autocorr) - 1
        
    if min_lag >= max_lag:
        return 0, autocorr
        
    # Find the peak in the valid lag range
    valid_autocorr = autocorr[min_lag:max_lag]
    if len(valid_autocorr) == 0:
        return 0, autocorr
        
    peak_lag = min_lag + np.argmax(valid_autocorr)
    
    # F0 = fs / lag
    f0 = sr / peak_lag if peak_lag > 0 else 0
    return f0, autocorr

def compute_fft_pitch(y, sr, fmin=75, fmax=500):
    """Computes F0 using the FFT technique for a single frame."""
    if len(y) == 0:
        return 0, np.array([]), np.array([])
        
    windowed_y = y * np.hanning(len(y))
    spectrum = np.abs(np.fft.rfft(windowed_y))
    freqs = np.fft.rfftfreq(len(windowed_y), 1/sr)
    
    # Restrict to valid F0 range
    valid_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if len(valid_indices) == 0:
        return 0, spectrum, freqs
        
    peak_idx = valid_indices[np.argmax(spectrum[valid_indices])]
    f0 = freqs[peak_idx]
    
    return f0, spectrum, freqs

def analyze_audio(y, sr, frame_ms=30, ste_threshold_ratio=0.1):
    """Performs windowing and computes F0 track."""
    frame_length = int(sr * frame_ms / 1000)
    hop_length = frame_length // 2  # 50% overlap
    
    if frame_length == 0 or len(y) < frame_length:
         return 0, 0, 0, []
         
    ste, zcr = calculate_ste_zcr(y, frame_length, hop_length)
    if len(ste) == 0:
        return 0, 0, 0, []
        
    # Define dynamic threshold based on max energy
    thresh = np.max(ste) * ste_threshold_ratio
    voiced_indices = np.where(ste > thresh)[0]
    
    if len(voiced_indices) == 0:
        return 0, np.mean(zcr), np.mean(ste), []
        
    f0s_autocorr = []
    
    # For actual length of y mapping to frames
    for idx in voiced_indices:
        start_sample = idx * hop_length
        end_sample = min(len(y), start_sample + frame_length)
        frame = y[start_sample:end_sample]
        
        f0_ac, _ = compute_autocorrelation_pitch(frame, sr)
        if f0_ac > 0:
            f0s_autocorr.append(f0_ac)
            
    mean_f0 = np.mean(f0s_autocorr) if len(f0s_autocorr) > 0 else 0
    mean_zcr = np.mean(zcr)
    mean_ste = np.mean(ste)
    
    return mean_f0, mean_zcr, mean_ste, f0s_autocorr
