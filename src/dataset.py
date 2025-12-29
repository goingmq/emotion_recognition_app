import os
import glob
import numpy as np
from .features import extract_features, extract_emotion_from_filename
from .config import DATA_PATH


def load_dataset():
    search = os.path.join(str(DATA_PATH), "**", "*.wav")
    files = glob.glob(search, recursive=True)
    print(f"Found {len(files)} wav files.")

    X, Y = [], []

    for f in files:
        emo = extract_emotion_from_filename(f)
        if emo is None:
            continue

        feat = extract_features(f)
        if feat is None:
            continue

        X.append(feat)
        Y.append(emo)

    X = np.array(X)
    Y = np.array(Y)
    print("X shape:", X.shape)

    return X, Y
