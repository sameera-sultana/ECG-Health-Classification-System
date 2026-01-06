import numpy as np
from scipy.signal import butter, lfilter

def bandpass_filter(signal, lowcut=0.5, highcut=40, fs=360, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def extract_beats(signal, annotation, window_size=180):
    beats = []
    labels = []

    for i, sym in enumerate(annotation.symbol):
        if sym in ['N', 'V', 'S', 'F']:
            idx = annotation.sample[i]
            start = max(idx - window_size, 0)
            end = min(idx + window_size, len(signal))
            beat = signal[start:end]

            # Only keep beats of exact length
            if len(beat) == 2 * window_size:
                beats.append(beat)
                labels.append(sym)

    return np.array(beats), np.array(labels)

def preprocess_for_inference(signal, target_len=360):
    """
    Used when annotations are NOT available (Streamlit UI).
    """
    signal = bandpass_filter(signal)

    if len(signal) >= target_len:
        signal = signal[:target_len]
    else:
        pad = target_len - len(signal)
        signal = np.pad(signal, (0, pad))

    # Avoid division by zero
    std = np.std(signal)
    if std == 0:
        std = 1e-6

    signal = (signal - np.mean(signal)) / std

    return signal.reshape(1, target_len, 1)
