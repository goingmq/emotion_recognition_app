import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.features import extract_features, pad_or_truncate
from src.dataset import load_dataset
from src.models import build_cnn_model

# 训练配置
DATA_PATH = "D:/语音识别/projects/emotion_recognition_app/data"  # 如果你有自己的数据路径就改这里
MODEL_OUT = "./ravdess_ser_cnn.h5"
TEST_SIZE = 0.2
EPOCHS = 30
BATCH_SIZE = 32
RANDOM_STATE = 42

# -------- 1. 加载数据 --------
print("Loading dataset...")
X, Y = load_dataset(DATA_PATH)

print("Dataset loaded! Shape:", X.shape)

# CNN reshape
X = X[..., np.newaxis]

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(Y)
y_onehot = tf.keras.utils.to_categorical(y_encoded)

num_classes = len(le.classes_)
print("Classes:", le.classes_)

# -------- 2. 划分数据集 --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------- 3. 构建 CNN 模型 --------
input_shape = (X.shape[1], X.shape[2], 1)
model = build_cnn_model(input_shape, num_classes)
model.summary()

# -------- 4. 训练并保存模型 --------
cb = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_OUT,
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )
]

print("Training CNN model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=cb,
    verbose=1
)

# -------- 5. 测试 --------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}")

print("\n=== Training Finished ===")
print("Model saved to:", MODEL_OUT)
