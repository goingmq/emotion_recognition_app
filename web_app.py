import os
import numpy as np
import librosa
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model

app = Flask(__name__)

# 静态上传目录（用于播放音频）
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "ravdess_ser_cnn.h5"

print(f"Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
class_names = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
print("Class names:", class_names)


# -------------------------------
# 音频特征提取函数：与训练一致
# -------------------------------
def extract_features(file_path, sr=22050, n_mfcc=40, max_pad_len=174):
    try:
        y, sr = librosa.load(file_path, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        stacked = np.vstack([mfcc, delta, delta2])

        if stacked.shape[1] < max_pad_len:
            pad_width = max_pad_len - stacked.shape[1]
            stacked = np.pad(stacked, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            stacked = stacked[:, :max_pad_len]

        return stacked
    except Exception as e:
        print(f"Audio load error: {e}")
        return None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        file = request.files.get("audio")

        if not file or file.filename == "":
            return render_template("index.html", error="请上传有效的音频文件！")

        filename = file.filename

        # 保存到 static/uploads 下方便前端播放
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # 提取特征
        features = extract_features(save_path)
        if features is None:
            return render_template("index.html", error="无法处理该音频文件，请使用 WAV 格式。")

        X = np.expand_dims(features, axis=(0, -1))

        # 模型预测
        probs = model.predict(X)[0]
        predicted_index = np.argmax(probs)
        predicted_label = class_names[predicted_index]

        prob_pairs = list(zip(class_names, probs.tolist()))

        # 返回并显示文件名 + 音频播放器
        return render_template(
            "index.html",
            result=predicted_label,
            prob_pairs=prob_pairs,
            filename=filename,
            audio_path=f"/static/uploads/{filename}",  # 让前端拿到可播放链接
            success=True
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
