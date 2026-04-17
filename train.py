import numpy as np
import sys
import os

sys.path.append(os.path.abspath("."))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from utils.dataset import load_ecg_data
from utils.preprocessing import preprocess_for_inference
from utils.evaluation import evaluate_model
from models.cnn_lstm_model import build_cnn_lstm


# ----------------------------
# Load dataset
# ----------------------------
record_ids = [
    '100','101','102','103','104','105','106','107','108','109',
    '111','112','113','114','115','116','117','118','119',
    '121','122','123','124',
    '200','201','202','203','205','207','208','209',
    '210','212','213','214','215','217','219',
    '220','221','222','223','228','230','231','232','233','234'
]

all_X, all_y = [], []

for record_id in record_ids:
    print(f"Loading record {record_id}...")
    X_rec, y_rec = load_ecg_data([record_id])

    for x_beat, y_label in zip(X_rec, y_rec):
        processed = preprocess_for_inference(x_beat.flatten())

        # ensure shape (360, 1)
        all_X.append(processed.squeeze())
        all_y.append(y_label)


X = np.array(all_X)
y = np.array(all_y)

print("Initial dataset shape:", X.shape, y.shape)


# ----------------------------
# Data augmentation (minority classes)
# ----------------------------
def augment_signal(beat):
    noise = np.random.normal(0, 0.01, beat.shape)
    return beat + noise


X_aug, y_aug = [], []

for xi, yi in zip(X, y):
    if yi in ['V', 'S', 'F']:   # minority classes
        X_aug.append(augment_signal(xi))
        y_aug.append(yi)

if len(X_aug) > 0:
    X = np.concatenate([X, np.array(X_aug)], axis=0)
    y = np.concatenate([y, np.array(y_aug)], axis=0)

print("After augmentation:", X.shape)


# ----------------------------
# Encode labels
# ----------------------------
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

joblib.dump(encoder, "models/label_encoder.pkl")

y_cat = to_categorical(y_enc)


# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_enc
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ----------------------------
# Build model
# ----------------------------
model = build_cnn_lstm((360, 1), y_cat.shape[1])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ----------------------------
# Class weights 
# ----------------------------
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_enc),
    y=y_enc
)

class_weights = {i: w for i, w in enumerate(class_weights_array)}


# ----------------------------
# Early stopping
# ----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


# ----------------------------
# Train model
# ----------------------------
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    class_weight=class_weights,
    callbacks=[early_stop]
)


# ----------------------------
# Evaluate model
# ----------------------------
print(evaluate_model(model, X_test, y_test))

preds = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(
    np.argmax(y_test, axis=1),
    np.argmax(preds, axis=1),
    target_names=encoder.classes_
))


# ----------------------------
# Save model
# ----------------------------
model.save("models/ecg_cnn_lstm.h5")

print("Model saved successfully ✔️")