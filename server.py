from flask import Flask, request, render_template, jsonify
import os
from src.predict import predict_emotion_from_file

app = Flask(__name__,
            template_folder="web/templates",
            static_folder="web/static")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files["audio"]
    filepath = "temp.wav"
    file.save(filepath)

    emotion, prob = predict_emotion_from_file(filepath)
    return jsonify({"emotion": emotion, "probabilities": prob})

if __name__ == "__main__":
    app.run(debug=True)
