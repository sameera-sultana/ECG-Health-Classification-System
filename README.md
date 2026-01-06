🫀 ECG Health Signal Classification using Deep Learning (CNN-LSTM)

📌 Overview

This project detects ECG heart abnormalities using a CNN-LSTM deep learning model.
It classifies ECG signals into Normal and Arrhythmia categories and provides prediction confidence.

🧠 Technologies

Python

TensorFlow / Keras

NumPy, SciPy

Scikit-learn

WFDB

Streamlit

🫀 Dataset

MIT-BIH Arrhythmia Dataset

ECG classes: N, V, S, F

🏗️ Project Structure
ECG_PROJECT/
├── models/
├── utils/
├── app.py
├── train.py
├── predict.py
└── requirements.txt

⚙️ Setup
pip install -r requirements.txt

🚀 Run Application
streamlit run app.py