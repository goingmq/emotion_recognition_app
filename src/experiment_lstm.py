import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from .dataset import load_dataset
from .models_lstm import build_lstm_model
from .utils import plot_confusion_matrix
from .config import TEST_SIZE, RANDOM_STATE, EPOCHS, BATCH_SIZE


def run_lstm_experiment():
    print("\nLoading dataset for LSTM...")
    X, Y = load_dataset()

    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)
    num_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y_encoded)

    # reshape for RNN format
    X_rnn = np.transpose(X, (0, 2, 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_rnn, y_cat,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes)
    model.summary()

    cb = [
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )

    # evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    preds = model.predict(X_test)
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, le.classes_, title="LSTM Confusion Matrix")

    # training curves
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("LSTM Training Loss")
    plt.legend()
    plt.show()
