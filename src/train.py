import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from .dataset import load_dataset
from .model import build_cnn_model
from .utils import plot_confusion_matrix
from .config import TEST_SIZE, RANDOM_STATE, BATCH_SIZE, EPOCHS


def train():
    print("Loading dataset...")
    X, Y = load_dataset()

    X = X[..., np.newaxis]

    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)
    num_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    model = build_cnn_model((X.shape[1], X.shape[2]), num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, le.classes_, title="CNN Confusion Matrix")

    from .config import MODEL_OUT
    model.save(MODEL_OUT)
    print("Model saved to", MODEL_OUT)
