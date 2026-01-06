import wfdb
import numpy as np
from utils.preprocessing import bandpass_filter, extract_beats

def load_ecg_data(record_ids, fs=360):
    all_X = []
    all_y = []

    for record_id in record_ids:
        print(f"Loading record {record_id}...")

        record = wfdb.rdrecord(record_id, pn_dir='mitdb')
        annotation = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')

        signal = record.p_signal[:, 0]
        filtered_signal = bandpass_filter(signal, fs=fs)

        X, y = extract_beats(filtered_signal, annotation)
        all_X.append(X)
        all_y.append(y)

    X = np.vstack(all_X)
    y = np.hstack(all_y)

    return X, y
