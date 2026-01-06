import numpy as np
import tensorflow as tf
import joblib

# Load trained model
MODEL_PATH = "models/ecg_cnn_lstm.h5"
model = tf.keras.models.load_model(MODEL_PATH)

ENCODER_PATH = "models/label_encoder.pkl"
encoder = joblib.load(ENCODER_PATH)
# Generate medical report
def generate_report(pred_class_symbol):
    reports = {
        'N': "Normal ECG detected.",
        'V': "Ventricular arrhythmia detected.",
        'S': "Supraventricular abnormality detected.",
        'F': "Fusion beat detected."
    }
    return reports.get(pred_class_symbol, "Unknown ECG condition detected.")

def predict_ecg(processed_signal):
    """
    processed_signal: shape (1, 360, 1), normalized
    Returns:
        pred_class_symbol (str)
        confidence (float)
        report (str)
        probs (dict) -> class-wise probabilities
    """

    # Ensure batch dimension
    if processed_signal.ndim == 2:
        processed_signal = processed_signal[np.newaxis, ...]

    preds = model.predict(processed_signal, verbose=0)

    pred_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    pred_class_symbol = encoder.inverse_transform([pred_idx])[0]

    # Class-wise probability dictionary
    probs = {
        cls: float(preds[0][i])
        for i, cls in enumerate(encoder.classes_)
    }

    report = generate_report(pred_class_symbol)

    return pred_class_symbol, confidence, report, probs
