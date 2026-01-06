from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, LSTM

def build_cnn_lstm(input_shape, num_classes):
    model = Sequential([
    Conv1D(128, 5, activation='relu', input_shape=input_shape),
    MaxPooling1D(2),
    LSTM(128),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')

    ])
    return model
