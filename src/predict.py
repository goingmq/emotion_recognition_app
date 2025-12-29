import numpy as np
import librosa
from .features import extract_features
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

MODEL_PATH = "./ravdess_ser_cnn.h5"

# 加载模型（程序启动时加载一次）
model = load_model(MODEL_PATH)

# 加载标签编码器（这些标签你应与训练时保持一致）
EMOTIONS = ['angry','calm','disgust','fearful','happy','neutral','sad','surprised']
le = LabelEncoder()
le.fit(EMOTIONS)

def predict_emotion_from_file(filepath):
    # 提取 MFCC
    features = extract_features(filepath)
    features = np.expand_dims(features, axis=0)  # (1, features, time)
    features = features[..., np.newaxis]  # (1, features, time, 1)

    # 模型预测
    preds = model.predict(features)
    emotion_id = np.argmax(preds)
    emotion = le.inverse_transform([emotion_id])[0]

    # 把概率转成 dict 返回给前端
    probabilities = {emo: float(preds[0][i]) for i, emo in enumerate(le.classes_)}

    return emotion, probabilities
