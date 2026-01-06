import numpy as np
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return classification_report(
        np.argmax(y_test, axis=1),
        np.argmax(preds, axis=1)
    )
