import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from .dataset import load_dataset
from .model import build_cnn_model
from .models_lstm import build_lstm_model
from .models_cnn_lstm import build_cnn_lstm_model
from .evaluation import evaluate_and_store_results
from .config import TEST_SIZE, RANDOM_STATE, EPOCHS, BATCH_SIZE


def run_comparison_experiments():
    print("\nLoading dataset for comparison experiments...")
    X, Y = load_dataset()

    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)
    num_classes = len(le.classes_)
    y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes)

    X_cnn = X[..., np.newaxis]
    X_lstm = np.transpose(X, (0, 2, 1))

    results = []
    histories = {}
    class_reports = {}

    cb = [tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]

    # =============== CNN ===============
    print("\nRunning CNN...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)

    cnn = build_cnn_model((X.shape[1], X.shape[2]), num_classes)
    histories["CNN"] = cnn.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
        callbacks=cb, verbose=0
    )
    evaluate_and_store_results(cnn, X_test, y_test, le, results, class_reports, "CNN")

    # =============== LSTM ===============
    print("Running LSTM...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_lstm, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)

    lstm = build_lstm_model((X_lstm.shape[1], X_lstm.shape[2]), num_classes)
    histories["LSTM"] = lstm.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
        callbacks=cb, verbose=0
    )
    evaluate_and_store_results(lstm, X_test, y_test, le, results, class_reports, "LSTM")

    # =============== CNN-LSTM ===============
    print("Running CNN-LSTM...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_cnn, y_cat, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded)

    cnn_lstm = build_cnn_lstm_model((X.shape[1], X.shape[2], 1), num_classes)
    histories["CNN-LSTM"] = cnn_lstm.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1,
        callbacks=cb, verbose=0
    )
    evaluate_and_store_results(cnn_lstm, X_test, y_test, le, results, class_reports, "CNN-LSTM")

    # --------- Visualization ---------
    df = pd.DataFrame(results)
    print("\n===== Final Comparison Table =====")
    print(df)

    # Plot performance comparison
    df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(10,6))
    sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric")
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.show()

    # Per-class F1 comparison
    per_class = []

    for model_name, report in class_reports.items():
        for emotion, data in report.items():
            if emotion in le.classes_:
                per_class.append({
                    "Model": model_name,
                    "Emotion": emotion,
                    "F1": data["f1-score"]
                })

    df_pc = pd.DataFrame(per_class)
    plt.figure(figsize=(12,6))
    sns.barplot(data=df_pc, x="Emotion", y="F1", hue="Model")
    plt.title("Per-Class F1 Score Comparison")
    plt.show()
