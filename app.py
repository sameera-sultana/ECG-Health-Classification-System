import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import wfdb

from utils.preprocessing import preprocess_for_inference, bandpass_filter
from predict import predict_ecg 

st.set_page_config(
    page_title="ECG Health Signal Classification",
    layout="centered"
)

st.title("🫀 ECG Health Signal Classification System")
st.write("Analyze ECG signals using a CNN-LSTM deep learning model.")

mode = st.radio(
    "Select ECG input method:",
    ["Use Sample ECG (MIT-BIH)", "Upload ECG File (.npy)"]
)

# OPTION 1: Sample ECG
if mode == "Use Sample ECG (MIT-BIH)":
    record_id = st.selectbox(
        "Select ECG Record ID",
        ["100", "101", "102", "103", "104",'111','112','113',
         '114','115','116','117','118','119', '121','122','123','124',
        '200','201','202','203','205','207','208','209',
        '210','212','213','214','215','217','219']
    )

    if st.button("📥 Load Sample ECG"):
        try:
            record = wfdb.rdrecord(record_id, pn_dir="mitdb")
            raw_signal = record.p_signal[:, 0]

            # Bandpass filter
            filtered_signal = bandpass_filter(raw_signal)

            st.subheader("📈 ECG Signal (Sample)")
            fig, ax = plt.subplots()
            ax.plot(filtered_signal[:1000])
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)

            # Store signal in session state
            st.session_state["signal"] = filtered_signal

        except Exception as e:
            st.error(f"Error loading record: {e}")

    if "signal" in st.session_state and st.button("🔍 Analyze ECG"):
        processed = preprocess_for_inference(st.session_state["signal"])

        pred_class, confidence, report, probs = predict_ecg(processed)

        st.subheader("🧠 Prediction Result")
        st.write(f"**Predicted Class:** {pred_class}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.info(report)

        st.subheader("📊 Class Probabilities")
        st.json(probs)

        if pred_class != 'N':
            st.warning("⚠ Abnormal ECG detected. Please consult a cardiologist.")
        else:
            st.success("✅ ECG appears normal.")
            
# OPTION 2: Upload ECG file
else:
    uploaded_file = st.file_uploader(
        "Upload ECG signal (.npy file)",
        type=["npy"]
    )

    if uploaded_file is not None:
        try:
            raw_signal = np.load(uploaded_file)
            filtered_signal = bandpass_filter(raw_signal)

            st.subheader("📈 Uploaded ECG Signal")
            fig, ax = plt.subplots()
            ax.plot(filtered_signal)
            ax.set_xlabel("Time")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)

            st.session_state["signal"] = filtered_signal

        except Exception as e:
            st.error(f"Error loading file: {e}")

    if "signal" in st.session_state and st.button("🔍 Analyze ECG"):
        processed = preprocess_for_inference(st.session_state["signal"])

        pred_class, confidence, report, probs = predict_ecg(processed)

        st.subheader("🧠 Prediction Result")
        st.write(f"**Predicted Class:** {pred_class}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.info(report)

        st.subheader("📊 Class Probabilities")
        st.json(probs)

        if pred_class != 'N':
            st.warning("⚠ Abnormal ECG detected. Please consult a cardiologist.")
        else:
            st.success("✅ ECG appears normal.")
